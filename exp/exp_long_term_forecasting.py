from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import scipy.interpolate as spi
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from PIL import Image
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, seasonal_decompose
warnings.filterwarnings('ignore')
from scipy.linalg import solve 
from scipy.interpolate import Rbf, splprep
from scipy.interpolate import lagrange
from scipy.interpolate import BarycentricInterpolator
from numpy.polynomial.polynomial import Polynomial 
import random

def rbf_smooth(Y): 
    X = torch.arange(1, Y.shape[1] + 1, 1)  
    new_x = torch.arange(1, Y.shape[1] + 1, 0.5)  
    new_x = new_x[:-1]
    YY = torch.zeros([Y.shape[0], Y.shape[1] * 2 - 1, Y.shape[2]])
    for i in range(Y.shape[0]):
        for j in range(Y.shape[2]):
            rf = Rbf(X.cpu(), Y[i, :, j].cpu())
            YY[i, :, j] = torch.from_numpy(rf(new_x.cpu()))
    return YY



def four_aver(x): 
    
    batch_size,T,F=x.shape
    interpolated_seq=torch.empty((batch_size, int(3/2 *T-1),F),dtype=x.dtype) 
    j = 0
    for i in range(0,T,2):
        interpolated_seq[:,j:j+2,:] = x[:,i:i+2,:]
        if i+3 < T:
            interpolated_seq[:,j+2,:] = (x[:,i,:]+x[:,i+1,:]+x[:,i+2,:]+x[:,i+3,:])/4
        j = j + 3
        
    return interpolated_seq

def smooth_seq(Y):
   
    X = torch.arange(1, Y.shape[1] + 1, 1) 
    new_x = torch.arange(1, Y.shape[1] + 1, 0.5)  
    new_x = new_x[:-1]
    YY = torch.zeros([Y.shape[0], Y.shape[1] * 2 - 1, Y.shape[2]])
    for i in range(Y.shape[0]):
        for j in range(Y.shape[2]):
            ipo = spi.splrep(X.cpu(), Y[i, :, j].cpu(), k=3)
            YY[i, :, j] = torch.from_numpy(spi.splev(new_x.cpu(), ipo))
    return YY
def smooth_sequence(x,a=0):
    
    batch_size,T,F=x.shape
    interpolated_seq=torch.empty((batch_size,2*T-1,F),dtype=x.dtype)
    for t in range(T-1):
        diff=(x[:,t+1,:]-x[:,t,:])/2
        interp=(x[:,t,:]+x[:,t+1,:])/2+a*diff
        interpolated_seq[:,2*t:2*t+2,:]=torch.stack([x[:,t,:],interp],axis=1)
    interpolated_seq[:,-1,:]=x[:,-1,:]
    return interpolated_seq

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.te = 0
        self.args = args
        self.functions_dict = {
            "rbf": rbf_smooth,
            "four_aver": four_aver,
            "spline": smooth_seq,
            "two_aver": smooth_sequence,
        }
        print("args.InterMethod",args.InterMethod)
        self.smooth = self.functions_dict[args.InterMethod]
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss() 
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if self.args.Interpolation == True:
                    batch_x = self.smooth(batch_x).float().to(self.device)
                    batch_x_mark = smooth_sequence(batch_x_mark,0).float().to(self.device)
                    if self.args.InterMethod == "four_aver":
                        xx = [i for i in range(batch_x_mark.shape[1]) if (i-1)%4!=0]
                        batch_x_mark = batch_x_mark[:,xx,:]
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            if epoch == 0:
                var1=[]
                var_rbf=[]
                var_two=[]
                var_four=[]
                var_spline=[]
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.Interpolation == True:    
                    batch_x = self.smooth(batch_x).float().to(self.device)
                    batch_x_mark = smooth_sequence(batch_x_mark,0).float().to(self.device)
                    if self.args.InterMethod == "four_aver":
                        xx = [i for i in range(batch_x_mark.shape[1]) if (i-1)%4!=0]
                        batch_x_mark = batch_x_mark[:,xx,:]
            
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        self.te = 1
        test_data, test_loader = self._get_data(flag='test') 
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('/home/kxg/gyy/Time-Series-Library-main/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if self.args.Interpolation == True: 
                    batch_x = self.smooth(batch_x).float().to(self.device)
                    batch_x_mark = smooth_sequence(batch_x_mark,0).float().to(self.device)
                    if self.args.InterMethod == "four_aver":
                        xx = [i for i in range(batch_x_mark.shape[1]) if (i-1)%4!=0]
                        batch_x_mark = batch_x_mark[:,xx,:]
                       
                    

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    """
                    seasonal_decomp_gt = seasonal_decompose(gt, model="additive",period=24)  
                    seasonal_decomp_pd = seasonal_decompose(pd, model="additive",period=24) 
                    plt.figure()
                    ax1 = plt.subplot(221)
                    plt.plot(seasonal_decomp_gt.trend,label='gt_trend') 
                    plt.plot(seasonal_decomp_pd.trend,label='pd_trend') 
                    ax2 = plt.subplot(222)
                    plt.plot(seasonal_decomp_gt.seasonal,label='gt_season') 
                    plt.plot(seasonal_decomp_pd.seasonal,label='pd_season') 
                    ax3 = plt.subplot(212)
                    plt.plot(gt, label='gt', linewidth=2)
                    plt.plot(pd, label='pd', linewidth=2)
                    ax3.legend()
                    plt.show()
                    plt.savefig(str(i) + '.png')  
                    """            
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return test_data.data_x,test_loader 
        # test_data.data_x: [len,N] 
