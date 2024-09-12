import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from einops import rearrange, repeat, reduce
import torch.nn.utils as U
from omegaconf import DictConfig
import opt_einsum as oe
from IPython import embed
from functools import partial
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def transition(measure, N, **measure_args):
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = 2*np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2
        B[0] = 2**.5
        A = A - B[:, None] * B[None, :]
        # A = A - np.eye(N)
        B *= 2**.5
        B = B[:, None]

    return A, B


def basis(method, N, vals, c=0.0, truncate_measure=True):
    """
    vals: list of times (forward in time)
    returns: shape (T, N) where T is length of vals
    """
    if method == 'legt':
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1).T
        eval_matrix *= (2 * np.arange(N) + 1) ** .5 * (-1) ** np.arange(N)
    elif method == 'legs':
        _vals = np.exp(-vals)
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * _vals).T  # (L, N)
        eval_matrix *= (2 * np.arange(N) + 1) ** .5 * (-1) ** np.arange(N)
    elif method == 'lagt':
        vals = vals[::-1]
        eval_matrix = ss.eval_genlaguerre(np.arange(N)[:, None], 0, vals)
        eval_matrix = eval_matrix * np.exp(-vals / 2)
        eval_matrix = eval_matrix.T
    elif method == 'fourier':
        cos = 2 ** .5 * np.cos(2 * np.pi * np.arange(N // 2)[:, None] * (vals))  # (N/2, T/dt)
        sin = 2 ** .5 * np.sin(2 * np.pi * np.arange(N // 2)[:, None] * (vals))  # (N/2, T/dt)
        cos[0] /= 2 ** .5
        eval_matrix = np.stack([cos.T, sin.T], axis=-1).reshape(-1, N)  # (T/dt, N)
    #     print("eval_matrix shape", eval_matrix.shape)

    if truncate_measure:
        eval_matrix[measure(method)(vals) == 0.0] = 0.0

    p = torch.tensor(eval_matrix)
    p *= np.exp(-c * vals)[:, None]  # [::-1, None]
    return p


def measure(method, c=0.0):
    if method == 'legt':
        fn = lambda x: np.heaviside(x, 0.0) * np.heaviside(1.0-x, 0.0)
    elif method == 'legs':
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method == 'lagt':
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method in ['fourier']:
        fn = lambda x: np.heaviside(x, 1.0) * np.heaviside(1.0-x, 1.0)
    else: raise NotImplementedError
    fn_tilted = lambda x: np.exp(c*x) * fn(x)
    return fn_tilted


class HiPPO(nn.Module):
    """ Linear time invariant x' = Ax + Bu """

    def __init__(self, N, seq_len, method='legt', dt=1.0, T=1.0, discretization='bilinear',
                 scale=False, c=0.0, fast=True):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.method = method
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c
        self.pred_len = int(1/dt)
        self.fast = fast

        A, B = transition(method, N)
        A = A + np.eye(N) * c
        self.A = A
        self.B = B.squeeze(-1)
        self.measure_fn = measure(method)

        C = np.ones((1, N))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)
        dB = dB.squeeze(-1)

        self.register_buffer('dA', torch.Tensor(dA))  # (N, N)
        self.register_buffer('dB', torch.Tensor(dB))  # (N,)
        if fast:
            Ks = []
            AB = self.dB
            A = self.dA
            for i in range(0, seq_len):
                AB = A @ AB
                Ks += [AB]
            K = torch.stack(Ks).transpose(-1, -2)
            self.register_buffer('K', torch.Tensor(K))

        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = basis(self.method, self.N, self.vals, c=self.c).to(device).to(torch.float32)  # (T/dt, N)
        self.measure = measure(self.method)(self.vals)

    def project(self, inputs):
        if self.fast:
            k = self.K
            T = inputs.shape[-1]
            k_f = torch.fft.rfft(k, n=2 * T)  # (H L)
            # print(k_f.shape)
            u_f = torch.fft.rfft(inputs, n=2 * T)  # (B H L)
            y_f = torch.einsum('bedl,nl->bednl', u_f, k_f)
            y = torch.fft.irfft(y_f, n=2 * T)[..., :T]
            y = y[..., -1]
            return y

        # inputs = rearrange(inputs, 'b l e -> l b e')
        # """
        # inputs : (length, ...)
        # output : (length, ..., N) where N is the order of the HiPPO projection
        # """
        # inputs = inputs.unsqueeze(-1)
        # u = inputs * self.dB  # (length, ..., N)
        # c = torch.zeros(u.shape[1:]).to(inputs)
        # cs = []
        # for f in inputs:
        #     c = F.linear(c, self.dA) + self.dB * f
        #     cs.append(c)
        # out = torch.stack(cs, dim=0)
        # out = rearrange(out, 'l b e n -> b l e n')
        # return out

    def reconstruct(self, c, evals=None):  # TODO take in a times array for reconstruction
        """
        c: (..., N,) HiPPO coefficients (same as x(t) in S4 notation)
        output: (..., L,)
        """
        eval_matrix = self.eval_matrix
        y = c @ (eval_matrix.T)
        return y
    
optimized = True

if optimized: # 1
    contract = oe.contract
else:
    contract = torch.einsum

def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer

class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude

class Modrelu(modrelu):
    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

class TransposedLinear(nn.Module):
    """ Linear module on the second-to-last dimension
    Assumes shape (B, D, L), where L can be 1 or more axis
    """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        # nn.Linear default init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
            setattr(self.bias, "_optim", {"weight_decay": 0.0})
        else:
            self.bias = 0.0

    def forward(self, x):
        num_axis = len(x.shape[2:])  # num_axis in L, for broadcasting bias
        y = contract('b u ..., v u -> b v ...', x, self.weight) + \
            self.bias.view(-1, *[1]*num_axis)
        return y


class TransposedLN(nn.Module):
    """ LayerNorm module over second dimension
    Assumes shape (B, D, L), where L can be 1 or more axis

    This is slow and a dedicated CUDA/Triton implementation shuld provide substantial end-to-end speedup
    """

    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay": 0.0})
            setattr(self.s, "_optim", {"weight_decay": 0.0})
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            # calc. stats over D dim / channels
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s/s) * (x-m+self.m)
        else:
            # move channel to last axis, apply layer_norm, then move channel back to second axis
            _x = self.ln(rearrange(x, 'b d ... -> b ... d'))
            y = rearrange(_x, 'b ... d -> b d ...')
        return y


def Activation(activation=None, size=None, dim=-1):
    if activation in [None, 'id', 'identity', 'linear']:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'modrelu':
        return Modrelu(size)
    elif activation == 'sqrelu':
        return SquaredReLU()
    elif activation == 'ln':
        return TransposedLN(dim)
    else:
        raise NotImplementedError(
            "hidden activation '{}' is not implemented".format(activation))


def LinearActivation(
    d_input, d_output, bias=True,
    zero_bias_init=False,
    transposed=False,
    initializer=None,
    activation=None,
    activate=False,  # Apply activation as part of this module
    weight_norm=False,
    **kwargs,
):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    # linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu':
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, d_output,
                                dim=1 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


class Normalization(nn.Module):
    def __init__(
        self,
        d,
        transposed=False, # Length dimension is -1 or -2
        _name_='layer',
        **kwargs
    ):
        super().__init__()
        self.transposed = transposed
        self._name_ = _name_

        if _name_ == 'layer':
            self.channel = True # Normalize over channel dimension
            if self.transposed:
                self.norm = TransposedLN(d, **kwargs)
            else:
                self.norm = nn.LayerNorm(d, **kwargs)
        elif _name_ == 'instance':
            self.channel = False
            norm_args = {'affine': False, 'track_running_stats': False}
            norm_args.update(kwargs)
            self.norm = nn.InstanceNorm1d(d, **norm_args) # (True, True) performs very poorly
        elif _name_ == 'batch':
            self.channel = False
            norm_args = {'affine': True, 'track_running_stats': True}
            norm_args.update(kwargs)
            self.norm = nn.BatchNorm1d(d, **norm_args)
        elif _name_ == 'group':
            self.channel = False
            self.norm = nn.GroupNorm(1, d, *kwargs)
        elif _name_ == 'none':
            self.channel = True
            self.norm = nn.Identity()
        else: raise NotImplementedError

    def forward(self, x):
        # Handle higher dimension logic
        shape = x.shape
        if self.transposed:
            x = rearrange(x, 'b d ... -> b d (...)')
        else:
            x = rearrange(x, 'b ... d -> b (...)d ')

        # The cases of LayerNorm / no normalization are automatically handled in all cases
        # Instance/Batch Norm work automatically with transposed axes
        if self.channel or self.transposed:
            x = self.norm(x)
        else:
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)

        x = x.view(shape)
        return x

    def step(self, x, **kwargs):
        assert self._name_ in ["layer", "none"]
        if self.transposed: x = x.unsqueeze(-1)
        x = self.forward(x)
        if self.transposed: x = x.squeeze(-1)
        return x


class GConv(nn.Module):#对照模型看看GConv实现了什么？才能决定用在iTransformer的哪个部分
    requires_length = True

    def __init__(
        self,batch_size,
        d_model,
        d_state=64,
        l_max=1,  # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
        channels=1,  # maps 1-dim to C-dim
        bidirectional=False,
        # Arguments for FF
        activation='gelu',  # activation in between SS and FF
        ln=False,  # Extra normalization
        postact=None,  # activation after FF
        initializer=None,  # initializer on FF
        weight_norm=False,  # weight normalization on FF
        hyper_act=None,  # Use a "hypernetwork" multiplication
        dropout=0.0,
        transposed=True,  # axis ordering (B, L, D) or (B, D, L)
        verbose=False,
        shift=False,
        linear=False,
        mode="cat_randn",
        # SSM Kernel arguments
        **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.ln = ln
        self.channels = channels # 传入n_heads
        self.transposed = transposed
        self.shift = shift
        self.linear = linear
        self.mode = mode
        self.l_max = l_max

        self.hyper = hyper_act is not None # 0
        if self.hyper:  # 0 
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional: # 1
            channels *= 2 # channels = 2 * n_heads

        # Pointwise
        if not self.linear:
            self.activation = Activation(activation)
            dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
            self.dropout = dropout_fn(
                dropout) if dropout > 0.0 else nn.Identity()
            if self.ln:
                self.norm = Normalization(
                    self.h*self.channels, transposed=transposed)
            else:
                self.norm = nn.Identity()

        # position-wise output transform to mix features
        if not self.linear:
            self.output_linear = LinearActivation(
                self.h*self.channels,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )

        self.init_scale = kernel_args.get('init_scale', 0)
        self.kernel_dim = kernel_args.get('kernel_dim', 64)
        self.num_scales = kernel_args.get(
            'n_scales', 1+math.ceil(math.log2(l_max/self.kernel_dim))-self.init_scale)
        if self.num_scales is None: # 1
            # l_max 传入seq_len，kernel_dim 传入32，init_scale取默认0
            self.num_scales = 1 + \
                math.ceil(math.log2(l_max/self.kernel_dim)) - self.init_scale
        self.kernel_list = nn.ParameterList()

        decay_min = kernel_args.get('decay_min', 2)
        decay_max = kernel_args.get('decay_max', 2)

        for _ in range(self.num_scales):  #以输入96为例， num_scales = 1+2-0=3
            if 'randn' in mode: # 1
                # channels = 2*n_heads self.h = enc_in self.kernel_dim = 32
                kernel = nn.Parameter(torch.randn(
                    channels, self.h, self.kernel_dim))
            elif 'cos' in mode:
                kernel = nn.Parameter(torch.cat([torch.cos(torch.linspace(0, 2*i*math.pi, self.kernel_dim)).expand(
                    channels, 1, self.kernel_dim) for i in range(self.h)], dim=1)[:, torch.randperm(self.h), :])
            else:
                raise ValueError(f"Unknown mode {mode}")
            kernel._optim = {
                'lr': kernel_args.get('lr', 0.001),
            }
            self.kernel_list.append(kernel) # [3,channels,enc_in,kernel_dim] 

        if 'learnable' in mode:
            self.decay = nn.Parameter(torch.rand(
                self.h) * (decay_max - decay_min) + decay_min)
            if 'fixed' in mode:
                self.decay.requires_grad = False
            else:
                self.decay._optim = {
                    'lr': kernel_args.get('lr', 0.001),
                }
            self.register_buffer('multiplier', torch.tensor(1.0))
        else: # 1
            # self.register_buffer不会有梯度传播给它，但是能被模型的state_dict记录下来。可以理解为模型的常数。
            # 这里的decay_min, decay_max都是2 self.h = enc_in
            # torch.linspace(start,end,step)
            """
            print(torch.linspace(3,10,5))
            结果：tensor([ 3.0000, 4.7500, 6.5000, 8.2500, 10.0000])
            torch.linspace(
                decay_min, decay_max, self.h).view(1, -1, 1) 返回[1,enc_in,1] 内容全是2
            """
            self.register_buffer('multiplier', torch.linspace(
                decay_min, decay_max, self.h).view(1, -1, 1))

        self.register_buffer('kernel_norm', torch.ones(channels, self.h, 1))
        self.register_buffer('kernel_norm_initialized',
                             torch.tensor(0, dtype=torch.bool))
        
        self.weights_real = nn.Parameter(
            torch.rand(batch_size, self.l_max, self.h, dtype=torch.float))
        self.weights_imag = nn.Parameter(
            torch.rand(batch_size, self.l_max, self.h, dtype=torch.float))
        self.D_real = nn.Parameter(torch.randn(channels, self.h))
        self.D_imag = nn.Parameter(torch.randn(channels, self.h))

    def forward(self, u, return_kernel=False): 
        """
        这里的H应该是enc_in ，L是input_len
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: # 1
            u = u.transpose(-1, -2) 
        L = u.size(-1) # u: [B,H,L]
        
        
        kernel_list = []
        interpolate_mode = 'nearest' if 'nearest' in self.mode else 'linear'  # 'linear'
        multiplier = self.multiplier
        if 'sum' in self.mode:
            for i in range(self.num_scales): 
                kernel = F.pad(
                    F.interpolate(
                        self.kernel_list[i],
                        scale_factor=2**(i+self.init_scale),
                        mode=interpolate_mode,
                    ),
                    (0, self.kernel_dim*2**(self.num_scales-1+self.init_scale) -
                     self.kernel_dim*2**(i+self.init_scale)),
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = sum(kernel_list)
        elif 'cat' in self.mode:   # 1
            """
                l_max 传入seq_len，self.kernel_dim 传入32，init_scale取默认0
                self.num_scales = 1 + math.ceil(math.log2(l_max/self.kernel_dim))
            """
            for i in range(self.num_scales):
                # F.interpolate 实现插值和上采样
                # mode表示使用的上采样算法 此处为linear
                """
                当输入长度为96
                这里的kernel_list为:  [3,channels,enc_in,kernel_dim] 
                scale_factor = 2**(max(0, i-1)+self.init_scale) = 1,1,2
                F.interpolate: 当输入scale_factor=1时， 输出等于输入
                否则，[channels,enc_in,kernel_dim] -> [channels,enc_in,kernel_dim*scale_factor] channels与n_heads有关
                举个例子：
                输入：tensor([[[ 1.,  2.,  3.,  4.],
                        [ 5.,  6.,  7.,  8.]],

                        [[ 0.,  2.,  4.,  6.],
                        [ 1.,  3.,  5.,  7.]],

                        [[ 3.,  4.,  5.,  6.],
                        [ 7.,  8.,  9., 10.]]])
                输出：[[[ 1.0000,  1.2500,  1.7500,  2.2500,  2.7500,  3.2500,  3.7500,  4.0000],
                        [ 5.0000,  5.2500,  5.7500,  6.2500,  6.7500,  7.2500,  7.7500,  8.0000]],

                       [[ 0.0000,  0.5000,  1.5000,  2.5000,  3.5000,  4.5000,  5.5000,  6.0000],
                        [ 1.0000,  1.5000,  2.5000,  3.5000,  4.5000,  5.5000,  6.5000,  7.0000]],

                        [[ 3.0000,  3.2500,  3.7500,  4.2500,  4.7500,  5.2500,  5.7500,  6.0000],
                        [ 7.0000,  7.2500,  7.7500,  8.2500,  8.7500,  9.2500,  9.7500,  10.0000]]]
                        multiplier: [1,enc_in,1] 内容全是2
                """
                # O(logL)个参数量相同的卷积核
                kernel = F.interpolate(
                    self.kernel_list[i],
                    scale_factor=2**(max(0, i-1)+self.init_scale),
                    mode=interpolate_mode,
                ) * multiplier ** (self.num_scales - i - 1)  # ** (self.num_scales - i - 1) 表示衰减，参考S4矩阵的幂次
                #  [channels,enc_in,kernel_dim*scale_factor] * [1,enc_in,1] -> [channels,enc_in,kernel_dim*scale_factor]
                # i=0时，(self.num_scales - i - 1) =2；i=1,(self.num_scales - i - 1)=1; i=2,(self.num_scales - i - 1)=0
                kernel_list.append(kernel) # 不完全正确 [3,channels,enc_in,kernel_dim*scale_factor]
            k = torch.cat(kernel_list, dim=-1) 
            # [channels,enc_in,kernel_dim*(scale_factor0+scale_factor1+scale_factor2)] (scale_factor0+scale_factor1+scale_factor2)=4
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if 'learnable' in self.mode: # 0
            k = k * torch.exp(-self.decay.view(1, -1, 1)*torch.log(
                torch.arange(k.size(-1), device=k.device)+1).view(1, 1, -1))

        if not self.kernel_norm_initialized: # 1
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach() # [channels,enc_in,1] 求欧几里得范数
            self.kernel_norm_initialized = torch.tensor(
                1, dtype=torch.bool, device=k.device)
            print(f"Kernel norm: {self.kernel_norm.mean()}")
            print(f"Kernel size: {k.size()}")

        if k.size(-1) > L: # u: [B,H,L]
            k = k[..., :L]
        elif k.size(-1) < L:
            k = F.pad(k, (0, L - k.size(-1)))
        # k: [channels,enc_in,L]
        k = k / self.kernel_norm  # * (L / self.l_max) ** 0.5

        # Convolution
        if self.bidirectional: # 1
            # rearrange 方法 k: [channels,enc_in,L]
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2) # k0:[n_heads,enc_in,L]
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0)) # [n_heads,enc_in,2L]
            # F.pad(k0, (0, L)) : (0, L)分别表示左边填充数和右边填充数 [n_heads,enc_in,2L]
            # F.pad(k1.flip(-1), (L, 0)) : 
            """
            flip(-1)效果：
            k1 tensor([[[ 3.,  4.,  5.,  6.],
                        [ 7.,  8.,  9., 10.]],

                    [[ 4.,  5.,  6.,  7.],
                    [ 8.,  9., 10.,  0.]]])
            k1.flip(-1) tensor([[[ 6.,  5.,  4.,  3.],
                                [10.,  9.,  8.,  7.]],

                                [[ 7.,  6.,  5.,  4.],
                                [ 0., 10.,  9.,  8.]]])
            """
        
        #k：  [n_heads,enc_in,2L]    u:[B,enc_in,L]
        k_f = torch.fft.rfft(k, n=2*L)  # (C H L)  n给定值，rfft输出最后一维维度为n/2+1
        u_f = torch.fft.rfft(u, n=2*L)  # (B H L)
        # print("k_f.shape",k_f.shape)
        # print("u_f.shape",u_f.shape)
        # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        
        y_f = contract('bhl,chl->bchl', u_f, k_f) 
        # contract=opt_einsum.contract :A drop in replacement for NumPy’s einsum function
        """
        einsum方法详细介绍
        ->左边和右边相同，把+=变成=即可
        https://blog.csdn.net/zhaohongfei_358/article/details/125273126
        y_f[bchl]=u_f[bhl]*k_f[chl]
        
        """
        y = torch.fft.irfft(y_f, n=2*L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.linear:
            y = self.dropout(self.activation(y))

        if not self.transposed: # 1
            y = y.transpose(-1, -2)

        if not self.linear:
            y = self.norm(y)
            y = self.output_linear(y)

        return y
    
    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)

class FNO(nn.Module):
    def __init__(self, in_channels, out_channels, enc_in, modes=32):
        super(FNO, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(scale * torch.rand(enc_in, in_channels,
                                                       out_channels, modes, dtype=torch.cfloat).to(device))

    def forward(self, x):
        B, E, L, D = x.shape  # shape=(batch, num_seq,seq_length, hidden_dim)
        x = rearrange(x, 'b e l d -> b e d l')
        #pdb.set_trace()
        x_f = torch.fft.rfft(x, n=2*L)
        k_f = self.weights
        y_f = torch.einsum('bedm,edcm->becm', x_f[..., :self.modes], k_f)
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]
        return y


class Film(nn.Module):
    def __init__(self, in_channels, out_channels, enc_in, seq_len, pred_len, N=128):
        super(Film, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.legt = HiPPO(method='legt', seq_len=seq_len, N=N, dt=1. / pred_len, fast=True)
        scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(scale * torch.rand(enc_in, in_channels,
                                                       out_channels, N).to(device))

    def forward(self, x):
        B, E, L, D = x.shape  # shape=(batch, num_seq,seq_length, hidden_dim)
        x = rearrange(x, 'b e l d -> b e d l')
        x_f = self.legt.project(x)
        k_f = self.weights
        y_f = torch.einsum('bedn,edcn->becn', x_f, k_f)
        y = self.legt.reconstruct(y_f)
        return y


if __name__ == '__main__':
    x = 7
    class Configs(object):
        ours = 0
        ab = 3
        # modes1 = 32
        seq_len = 192
        label_len = 96
        pred_len = 192
        output_attention = True
        enc_in = x
        dec_in = x
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        moving_avg = 25
        c_out = 1
        activation = 'gelu'
        wavelet = 0
        modes = 32
        N = 256

    configs = Configs()
    model1 = FNO(in_channels=1, out_channels=1, enc_in=configs.enc_in, modes=64)
    model2 = Film(in_channels=1, out_channels=1, enc_in=configs.enc_in, N=128,
                  seq_len=configs.seq_len, pred_len=configs.pred_len)

    enc = torch.randn([3, 1, configs.seq_len, configs.enc_in])
    out1 = model1.forward(enc)
    out2 = model2.forward(enc)
    assert out2.shape == out1.shape
    a = 1


