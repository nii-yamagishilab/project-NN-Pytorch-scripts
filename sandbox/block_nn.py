##!/usr/bin/env python
"""
Common blocks for neural networks

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
from scipy import signal as scipy_signal

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import torch.nn.init as torch_init

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


# For blstm
class BLSTMLayer(torch_nn.Module):
    """ Wrapper over dilated conv1D
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same
    """
    def __init__(self, input_dim, output_dim):
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = torch_nn.LSTM(input_dim, output_dim // 2, \
                                     bidirectional=True)
    def forward(self, x):
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)
        
#        
# 1D dilated convolution that keep the input/output length
class Conv1dKeepLength(torch_nn.Conv1d):
    """ Wrapper for causal convolution
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    https://github.com/pytorch/pytorch/issues/1333
    Note: Tanh is applied
    """
    def __init__(self, input_dim, output_dim, dilation_s, kernel_s, 
                 causal = False, stride = 1, groups=1, bias=True, \
                 tanh = True, pad_mode='constant'):
        super(Conv1dKeepLength, self).__init__(
            input_dim, output_dim, kernel_s, stride=stride,
            padding = 0, dilation = dilation_s, groups=groups, bias=bias)
        
        self.causal = causal
        # input & output length will be the same        
        if self.causal:
            # left pad to make the convolution causal
            self.pad_le = dilation_s * (kernel_s - 1)
            self.pad_ri = 0
        else:
            # pad on both sizes
            self.pad_le = dilation_s * (kernel_s - 1) // 2
            self.pad_ri = dilation_s * (kernel_s - 1) - self.pad_le

        # we may wrap other functions too
        if tanh:
            self.l_ac = torch_nn.Tanh()
        else:
            self.l_ac = torch_nn.Identity()

        self.pad_mode = pad_mode
        #
        return

    def forward(self, data):
        # permute to (batchsize=1, dim, length)
        # add one dimension (batchsize=1, dim, ADDED_DIM, length)
        # pad to ADDED_DIM
        # squeeze and return to (batchsize=1, dim, length)
        # https://github.com/pytorch/pytorch/issues/1333
        x = torch_nn_func.pad(data.permute(0, 2, 1).unsqueeze(2), \
                              (self.pad_le, self.pad_ri, 0, 0),
                              mode = self.pad_mode).squeeze(2)
        # tanh(conv1())
        # permmute back to (batchsize=1, length, dim)
        output = self.l_ac(super(Conv1dKeepLength, self).forward(x))
        return output.permute(0, 2, 1)

# 
# Moving average
class MovingAverage(Conv1dKeepLength):
    """ Wrapper to define a moving average smoothing layer
    Note: MovingAverage can be implemented using TimeInvFIRFilter too.
          Here we define another Module dicrectly on Conv1DKeepLength
    """
    def __init__(self, feature_dim, window_len, causal=False):
        super(MovingAverage, self).__init__(
            feature_dim, feature_dim, 1, window_len, causal,
            groups=feature_dim, bias=False, tanh=False)
        # set the weighting coefficients
        torch_nn.init.constant_(self.weight, 1/window_len)
        # turn off grad for this layer
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, data):
        return super(MovingAverage, self).forward(data)

# 
# FIR filter layer
class TimeInvFIRFilter(Conv1dKeepLength):                                    
    """ Wrapper to define a FIR filter over Conv1d
        Note: FIR Filtering is conducted on each dimension (channel)
        independently: groups=channel_num in conv1d
    """                                                                   
    def __init__(self, feature_dim, filter_coef, 
                 causal=True, flag_train=False):
        """ __init__(self, feature_dim, filter_coef, 
                 causal=True, flag_train=False)
        feature_dim: dimension of input data
        filter_coef: 1-D tensor of filter coefficients
        causal: FIR is causal or not (default: true)
        flag_train: whether train the filter coefficients (default: false)

        Input data: (batchsize=1, length, feature_dim)
        Output data: (batchsize=1, length, feature_dim)
        """
        super(TimeInvFIRFilter, self).__init__(                              
            feature_dim, feature_dim, 1, filter_coef.shape[0], causal,
            groups=feature_dim, bias=False, tanh=False)
        
        if filter_coef.ndim == 1:
            # initialize weight using provided filter_coef
            with torch.no_grad():
                tmp_coef = torch.zeros([feature_dim, 1, 
                                        filter_coef.shape[0]])
                tmp_coef[:, 0, :] = filter_coef
                tmp_coef = torch.flip(tmp_coef, dims=[2])
                self.weight = torch.nn.Parameter(tmp_coef, 
                                                 requires_grad=flag_train)
        else:
            print("TimeInvFIRFilter expects filter_coef to be 1-D tensor")
            print("Please implement the code in __init__ if necessary")
            sys.exit(1)

    def forward(self, data):                                              
        return super(TimeInvFIRFilter, self).forward(data)    

# 
# Up sampling
class UpSampleLayer(torch_nn.Module):
    """ Wrapper over up-sampling
    Input tensor: (batchsize=1, length, dim)
    Ouput tensor: (batchsize=1, length * up-sampling_factor, dim)
    """
    def __init__(self, feature_dim, up_sampling_factor, smoothing=False):
        super(UpSampleLayer, self).__init__()
        # wrap a up_sampling layer
        self.scale_factor = up_sampling_factor
        self.l_upsamp = torch_nn.Upsample(scale_factor=self.scale_factor)
        if smoothing:
            self.l_ave1 = MovingAverage(feature_dim, self.scale_factor)
            self.l_ave2 = MovingAverage(feature_dim, self.scale_factor)
        else:
            self.l_ave1 = torch_nn.Identity()
            self.l_ave2 = torch_nn.Identity()
        return
    
    def forward(self, x):
        # permute to (batchsize=1, dim, length)
        up_sampled_data = self.l_upsamp(x.permute(0, 2, 1))
        # permute it backt to (batchsize=1, length, dim)
        # and do two moving average
        return self.l_ave1(self.l_ave2(up_sampled_data.permute(0, 2, 1)))


class upsampleByTransConv(torch_nn.Module):
    """upsampleByTransConv
    Upsampling layer using transposed convolution
    """
    def __init__(self, feat_dim, output_dim, upsample_rate, window_ratio=5):
        """upsampleByTransConv(feat_dim, upsample_rate, window_ratio=5)

        Args
        ----
          feat_dim: int, input feature should be (batch, length, feat_dim)
          upsample_rate, int, output feature will be
                (batch, length*upsample_rate, feat_dim)
          window_ratio: int, default 5, window length of transconv will be
                upsample_rate * window_ratio
        """
        super(upsampleByTransConv, self).__init__()
        window_l = upsample_rate * window_ratio
        self.m_layer = torch_nn.ConvTranspose1d(
            feat_dim, output_dim, window_l, stride=upsample_rate)
        self.m_uprate = upsample_rate
        return

    def forward(self, x):
        """ y = upsampleByTransConv(x)
        input
        -----
          x: tensor, (batch, length, feat_dim)
        output
        ------
          y: tensor, (batch, length*upsample_rate, output_dim)
        """
        l = x.shape[1] * self.m_uprate
        y = self.m_layer(x.permute(0, 2, 1))[:, :, 0:l]
        return y.permute(0, 2, 1).contiguous()


class TimeVarFIRFilter(torch_nn.Module):
    """ TimeVarFIRFilter
    Given sequences of filter coefficients and a signal, do filtering
    
    Filter coefs: (batchsize, signal_length, filter_order = K)
    Signal:       (batchsize, signal_length, 1)
    
    For batch 0:
     For n in [1, sequence_length):
       output(0, n, 1) = \sum_{k=1}^{K} signal(0, n-k, 1)*coef(0, n, k)
       
    Note: filter coef (0, n, :) is only used to compute the output 
          at (0, n, 1)
    """
    def __init__(self):
        super(TimeVarFIRFilter, self).__init__()
    
    def forward(self, signal, f_coef):
        """ 
        Filter coefs: (batchsize=1, signal_length, filter_order = K)
        Signal:       (batchsize=1, signal_length, 1)
        
        Output:       (batchsize=1, signal_length, 1)
        
        For n in [1, sequence_length):
         output(0, n, 1) = \sum_{k=1}^{K} signal(0, n-k, 1)*coef(0, n, k)
          
        This method may be not efficient:
        
        Suppose signal [x_1, ..., x_N], filter [a_1, ..., a_K]
        output         [y_1, y_2, y_3, ..., y_N, *, * ... *]
               = a_1 * [x_1, x_2, x_3, ..., x_N,   0, ...,   0]
               + a_2 * [  0, x_1, x_2, x_3, ..., x_N,   0, ...,  0]
               + a_3 * [  0,   0, x_1, x_2, x_3, ..., x_N, 0, ...,  0]
        """
        signal_l = signal.shape[1]
        order_k = f_coef.shape[-1]

        # pad to (batchsize=1, signal_length + filter_order-1, dim)
        padded_signal = torch_nn_func.pad(signal, (0, 0, 0, order_k - 1))
        
        y = torch.zeros_like(signal)
        # roll and weighted sum, only take [0:signal_length]
        for k in range(order_k):
            y += torch.roll(padded_signal, k, dims=1)[:, 0:signal_l, :] \
                      * f_coef[:, :, k:k+1] 
        # done
        return y

class SignalsConv1d(torch_nn.Module):
    """ Filtering input signal with time invariant filter
    Note: FIRFilter conducted filtering given fixed FIR weight
          SignalsConv1d convolves two signals
    Note: this is based on torch.nn.functional.conv1d
    
    """                                                                  
    def __init__(self):
        super(SignalsConv1d, self).__init__()
        
    def forward(self, signal, system_ir):
        """ output = forward(signal, system_ir)
        
        signal:    (batchsize, length1, dim)
        system_ir: (length2, dim) 
        
        output:    (batchsize, length1, dim)
        """ 
        if signal.shape[-1] != system_ir.shape[-1]:
            print("Error: SignalsConv1d expects shape:")
            print("signal    (batchsize, length1, dim)")
            print("system_id (batchsize, length2, dim)")
            print("But received signal: {:s}".format(str(signal.shape)))
            print(" system_ir: {:s}".format(str(system_ir.shape)))
            sys.exit(1)
        padding_length = system_ir.shape[0] - 1
        groups = signal.shape[-1]
        
        # pad signal on the left 
        signal_pad = torch_nn_func.pad(signal.permute(0, 2, 1),\
                                       (padding_length, 0))
        # prepare system impulse response as (dim, 1, length2)
        # also flip the impulse response
        ir = torch.flip(system_ir.unsqueeze(1).permute(2, 1, 0), \
                        dims=[2])
        # convolute
        output = torch_nn_func.conv1d(signal_pad, ir, groups=groups)
        return output.permute(0, 2, 1)


# Sinc filter generator
class SincFilter(torch_nn.Module):
    """ SincFilter
        Given the cut-off-frequency, produce the low-pass and high-pass
        windowed-sinc-filters.
        If input cut-off-frequency is (batchsize=1, signal_length, 1),
        output filter coef is (batchsize=1, signal_length, filter_order).
        For each time step in [1, signal_length), we calculate one
        filter for low-pass sinc filter and another for high-pass filter.
        
        Example:
        import scipy
        import scipy.signal
        import numpy as np
        
        filter_order = 31
        cut_f = 0.2
        sinc_layer = SincFilter(filter_order)
        lp_coef, hp_coef = sinc_layer(torch.ones(1, 10, 1) * cut_f)
        
        w, h1 = scipy.signal.freqz(lp_coef[0, 0, :].numpy(), [1])
        w, h2 = scipy.signal.freqz(hp_coef[0, 0, :].numpy(), [1])
        plt.plot(w, 20*np.log10(np.abs(h1)))
        plt.plot(w, 20*np.log10(np.abs(h2)))
        plt.plot([cut_f * np.pi, cut_f * np.pi], [-100, 0])
    """
    def __init__(self, filter_order):
        super(SincFilter, self).__init__()
        # Make the filter oder an odd number
        #  [-(M-1)/2, ... 0, (M-1)/2]
        # 
        self.half_k = (filter_order - 1) // 2
        self.order = self.half_k * 2 +1
        
    def hamming_w(self, n_index):
        """ prepare hamming window for each time step
        n_index (batchsize=1, signal_length, filter_order)
            For each step, n_index.shape is [-(M-1)/2, ... 0, (M-1)/2]
            where,
            n_index[0, 0, :] = [-(M-1)/2, ... 0, (M-1)/2]
            n_index[0, 1, :] = [-(M-1)/2, ... 0, (M-1)/2]
            ...
        output  (batchsize=1, signal_length, filter_order)
            output[0, 0, :] = hamming_window
            output[0, 1, :] = hamming_window
            ...
        """
        # Hamming window
        return 0.54 + 0.46 * torch.cos(2 * np.pi * n_index / self.order)
    
    def sinc(self, x):
        """ Normalized sinc-filter sin( pi * x) / pi * x
        https://en.wikipedia.org/wiki/Sinc_function
        
        Assume x (batchsize, signal_length, filter_order) and 
        x[0, 0, :] = [-half_order, - half_order+1, ... 0 ..., half_order]
        x[:, :, self.half_order] -> time index = 0, sinc(0)=1
        """
        y = torch.zeros_like(x)
        y[:,:,0:self.half_k]=torch.sin(np.pi * x[:, :, 0:self.half_k]) \
                              / (np.pi * x[:, :, 0:self.half_k])
        y[:,:,self.half_k+1:]=torch.sin(np.pi * x[:, :, self.half_k+1:])\
                               / (np.pi * x[:, :, self.half_k+1:])
        y[:,:,self.half_k] = 1
        return y
        
    def forward(self, cut_f):
        """ lp_coef, hp_coef = forward(self, cut_f)
        cut-off frequency cut_f (batchsize=1, length, dim = 1)
    
        lp_coef: low-pass filter coefs  (batchsize, length, filter_order)
        hp_coef: high-pass filter coefs (batchsize, length, filter_order)
        """
        # create the filter order index
        with torch.no_grad():   
            # [- (M-1) / 2, ..., 0, ..., (M-1)/2]
            lp_coef = torch.arange(-self.half_k, self.half_k + 1, 
                                   device=cut_f.device)
            # [[[- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   [- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   ...
            #  ],
            #  [[- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   [- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   ...
            #  ]]
            lp_coef = lp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1)
            
            hp_coef = torch.arange(-self.half_k, self.half_k + 1, 
                                   device=cut_f.device)
            hp_coef = hp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1)
            
            # temporary buffer of [-1^n] for gain norm in hp_coef
            tmp_one = torch.pow(-1, hp_coef)
            
        # unnormalized filter coefs with hamming window
        lp_coef = cut_f * self.sinc(cut_f * lp_coef) \
                  * self.hamming_w(lp_coef)
        
        hp_coef = (self.sinc(hp_coef) \
                   - cut_f * self.sinc(cut_f * hp_coef)) \
                  * self.hamming_w(hp_coef)
        
        # normalize the coef to make gain at 0/pi is 0 dB
        # sum_n lp_coef[n]
        lp_coef_norm = torch.sum(lp_coef, axis=2).unsqueeze(-1)
        # sum_n hp_coef[n] * -1^n
        hp_coef_norm = torch.sum(hp_coef * tmp_one, axis=2).unsqueeze(-1)
        
        lp_coef = lp_coef / lp_coef_norm
        hp_coef = hp_coef / hp_coef_norm
        
        # return normed coef
        return lp_coef, hp_coef

class BatchNorm1DWrapper(torch_nn.BatchNorm1d):
    """
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(BatchNorm1DWrapper, self).__init__(
        num_features, eps, momentum, affine, track_running_stats)

    def forward(self, data):
        output = super(BatchNorm1DWrapper, self).forward(data.permute(0, 2, 1))
        return output.permute(0, 2, 1)


class SignalFraming(torch_nn.Conv1d):
    """ SignalFraming(w_len, h_len, w_type='Hamming')
    
    Do framing on the signal. The implementation is based on conv1d
    
    Args:
    -----
      w_len: window length (frame length)
      h_len: hop length (frame shift)
      w_type: type of window, (default='Hamming')
         Hamming: Hamming window
         else: square window
    
    Note: 
    -----
      input signal (batch, length, 1)
      output signal (batch, frame_num, frame_length)
            
      where frame_num = length + (frame_length - frame_num)
    
      Compatibility with Librosa framing need to be checked
    """                                                                   
    def __init__(self, w_len, h_len, w_type='Hamming'):
        super(SignalFraming, self).__init__(1, w_len, w_len, stride=h_len,
                padding = 0, dilation = 1, groups=1, bias=False)
        self.m_wlen = w_len
        self.m_wtype = w_type
        self.m_hlen = h_len

        if w_type == 'Hamming':
            self.m_win = scipy_signal.windows.hamming(self.m_wlen)
        else:
            self.m_win = np.ones([self.m_wlen])
        
        # for padding
        if h_len > w_len:
            print("Error: SignalFraming(w_len, h_len)")
            print("w_len cannot be < h_len")
            sys.exit(1)
        self.m_mat = np.diag(self.m_win)
        self.m_pad_len_l = (w_len - h_len)//2
        self.m_pad_len_r = (w_len - h_len) - self.m_pad_len_l
        
        # filter [output_dim = frame_len, 1, input_dim=frame_len]
        # No need to flip the filter coefficients
        with torch.no_grad():
            tmp_coef = torch.zeros([w_len, 1, w_len])
            tmp_coef[:, 0, :] = torch.tensor(self.m_mat)
            self.weight = torch.nn.Parameter(tmp_coef, requires_grad = False)
        return
    
    def forward(self, signal):
        """ 
        signal:    (batchsize, length1, 1)
        output:    (batchsize, num_frame, frame_length)
        
        Note: 
        """ 
        if signal.shape[-1] > 1:
            print("Error: SignalFraming expects shape:")
            print("signal    (batchsize, length, 1)")
            sys.exit(1)

        # 1. switch dimension from (batch, length, dim) to (batch, dim, length)
        # 2. pad signal on the left to (batch, dim, length + pad_length)
        signal_pad = torch_nn_func.pad(signal.permute(0, 2, 1),\
                                       (self.m_pad_len_l, self.m_pad_len_r))

        # switch dimension from (batch, dim, length) to (batch, length, dim)
        return super(SignalFraming, self).forward(signal_pad).permute(0, 2, 1)


class Conv1dStride(torch_nn.Conv1d):
    """ Wrapper for normal 1D convolution with stride (optionally)
    
    Input tensor:  (batchsize, length, dim_in)
    Output tensor: (batchsize, length2, dim_out)
    
    However, we wish that length2 = floor(length / stride)
    Therefore, 
    padding_total_length - dilation_s * (kernel_s - 1) -1 + stride = 0 
    or, 
    padding_total_length = dilation_s * (kernel_s - 1) + 1 - stride
    
    Conv1dBundle(input_dim, output_dim, dilation_s, kernel_s, 
                 causal = False, stride = 1, groups=1, bias=True, \
                 tanh = True, pad_mode='constant')
                 
        input_dim: int, input dimension (input channel)
        output_dim: int, output dimension (output channel)
        kernel_s: int, kernel size of filter        
        dilation_s: int, dilation for convolution
        causal: bool, whether causal convolution, default False
        stride: int, stride size, default 1
        groups: int, group for conv1d, default 1
        bias: bool, whether add bias, default True
        tanh: bool, whether use tanh activation, default True
        pad_mode: str, padding method, default "constant"
    """
    def __init__(self, input_dim, output_dim, kernel_s, dilation_s=1, 
                 causal = False, stride = 1, groups=1, bias=True, \
                 tanh = True, pad_mode='constant'):
        super(Conv1dStride, self).__init__(
            input_dim, output_dim, kernel_s, stride=stride,
            padding = 0, dilation = dilation_s, groups=groups, bias=bias)

        self.pad_mode = pad_mode
        self.causal = causal
        
        # padding size
        # input & output length will be the same
        if self.causal:
            # left pad to make the convolution causal
            self.pad_le = dilation_s * (kernel_s - 1) + 1 - stride
            self.pad_ri = 0
        else:
            # pad on both sizes
            self.pad_le = (dilation_s*(kernel_s-1)+1-stride) // 2
            self.pad_ri = (dilation_s*(kernel_s-1)+1-stride) - self.pad_le
    
        # activation functions
        if tanh:
            self.l_ac = torch_nn.Tanh()
        else:
            self.l_ac = torch_nn.Identity()
        
    def forward(self, data):
        # https://github.com/pytorch/pytorch/issues/1333
        # permute to (batchsize=1, dim, length)
        # add one dimension as (batchsize=1, dim, ADDED_DIM, length)
        # pad to ADDED_DIM
        # squeeze and return to (batchsize=1, dim, length+pad_length)
        x = torch_nn_func.pad(
            data.permute(0, 2, 1).unsqueeze(2), \
            (self.pad_le, self.pad_ri,0,0), \
            mode = self.pad_mode).squeeze(2)
        
        # tanh(conv1())
        # permmute back to (batchsize=1, length, dim)
        output = self.l_ac(super(Conv1dStride, self).forward(x))
        return output.permute(0, 2, 1)

    
    
class MaxPool1dStride(torch_nn.MaxPool1d):
    """ Wrapper for maxpooling
    
    Input tensor:  (batchsize, length, dim_in)
    Output tensor: (batchsize, length2, dim_in)
    
    However, we wish that length2 = floor(length / stride)
    Therefore, 
    padding_total_length - dilation_s * (kernel_s - 1) -1 + stride = 0 
    or, 
    padding_total_length = dilation_s * (kernel_s - 1) + 1 - stride
    
    MaxPool1dStride(kernel_s, stride, dilation_s=1)
    """
    def __init__(self, kernel_s, stride, dilation_s=1):
        super(MaxPool1dStride, self).__init__(
            kernel_s, stride, 0, dilation_s)
        
        # pad on both sizes
        self.pad_le = (dilation_s*(kernel_s-1)+1-stride) // 2
        self.pad_ri = (dilation_s*(kernel_s-1)+1-stride) - self.pad_le
        
    def forward(self, data):
        # https://github.com/pytorch/pytorch/issues/1333
        # permute to (batchsize=1, dim, length)
        # add one dimension as (batchsize=1, dim, ADDED_DIM, length)
        # pad to ADDED_DIM
        # squeeze and return to (batchsize=1, dim, length+pad_length)
        x = torch_nn_func.pad(
            data.permute(0, 2, 1).unsqueeze(2), \
            (self.pad_le, self.pad_ri,0,0)).squeeze(2).contiguous()
        
        # tanh(conv1())
        # permmute back to (batchsize=1, length, dim)
        output = super(MaxPool1dStride, self).forward(x)
        return output.permute(0, 2, 1)

    
    
class AvePool1dStride(torch_nn.AvgPool1d):
    """ Wrapper for average pooling
    
    Input tensor:  (batchsize, length, dim_in)
    Output tensor: (batchsize, length2, dim_in)
    
    However, we wish that length2 = floor(length / stride)
    Therefore, 
    padding_total_length - dilation_s * (kernel_s - 1) -1 + stride = 0 
    or, 
    padding_total_length = dilation_s * (kernel_s - 1) + 1 - stride
    
    MaxPool1dStride(kernel_s, stride, dilation_s=1)
    """
    def __init__(self, kernel_s, stride):
        super(AvePool1dStride, self).__init__(
            kernel_s, stride, 0)
        
        # pad on both sizes
        self.pad_le = ((kernel_s-1)+1-stride) // 2
        self.pad_ri = ((kernel_s-1)+1-stride) - self.pad_le
        
    def forward(self, data):
        # https://github.com/pytorch/pytorch/issues/1333
        # permute to (batchsize=1, dim, length)
        # add one dimension as (batchsize=1, dim, ADDED_DIM, length)
        # pad to ADDED_DIM
        # squeeze and return to (batchsize=1, dim, length+pad_length)
        x = torch_nn_func.pad(
            data.permute(0, 2, 1).unsqueeze(2), \
            (self.pad_le, self.pad_ri,0,0)).squeeze(2).contiguous()
        
        # tanh(conv1())
        # permmute back to (batchsize=1, length, dim)
        output = super(AvePool1dStride, self).forward(x)
        return output.permute(0, 2, 1)



class Maxout1D(torch_nn.Module):
    """ Maxout activation (along 1D)
    Maxout(d_in, d_out, pool_size)
    From https://github.com/pytorch/pytorch/issues/805
    
    Arguments
    ---------
    d_in: feature input dimension
    d_out: feature output dimension
    pool_size: window size of max-pooling
    
    
    Usage
    -----
    l_maxout1d = Maxout1D(d_in, d_out, pool_size)
    data_in = torch.rand([1, T, d_in])
    data_out = l_maxout1d(data_in)
    """
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = torch_nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        # suppose inputs (batchsize, length, dim)
        
        # shape (batchsize, length, out-dim, pool_size)
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        # shape (batchsize, length, out-dim * pool_size)
        out = self.lin(inputs)
        # view to (batchsize, length, out-dim, pool_size)
        # maximize on the last dimension
        m, i = out.view(*shape).max(max_dim)
        return m



class MaxFeatureMap2D(torch_nn.Module):
    """ Max feature map (along 2D) 
    
    MaxFeatureMap2D(max_dim=1)
    
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)

    
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """
    def __init__(self, max_dim = 1):
        super().__init__()
        self.max_dim = max_dim
        
    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)
        
        shape = list(inputs.size())
        
        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim]//2
        shape.insert(self.max_dim, 2)
        
        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m


class SelfWeightedPooling(torch_nn.Module):
    """ SelfWeightedPooling module
    Inspired by
    https://github.com/joaomonteirof/e2e_antispoofing/blob/master/model.py
    To avoid confusion, I will call it self weighted pooling
    
    Using self-attention format, this is similar to softmax(Query, Key)Value
    where Query is a shared learnarble mm_weight, Key and Value are the input
    Sequence.

    l_selfpool = SelfWeightedPooling(5, 1, False)
    with torch.no_grad():
        input_data = torch.rand([3, 10, 5])
        output_data = l_selfpool(input_data)
    """
    def __init__(self, feature_dim, num_head=1, mean_only=False):
        """ SelfWeightedPooling(feature_dim, num_head=1, mean_only=False)
        Attention-based pooling
        
        input (batchsize, length, feature_dim) ->
        output 
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
        
        args
        ----
          feature_dim: dimension of input tensor
          num_head: number of heads of attention
          mean_only: whether compute mean or mean with std
                     False: output will be (batchsize, feature_dim*2)
                     True: output will be (batchsize, feature_dim)
        """
        super(SelfWeightedPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mean_only = mean_only
        self.noise_std = 1e-5
        self.num_head = num_head

        # transformation matrix (num_head, feature_dim)
        self.mm_weights = torch_nn.Parameter(
            torch.Tensor(num_head, feature_dim), requires_grad=True)
        torch_init.kaiming_uniform_(self.mm_weights)
        return
    
    def _forward(self, inputs):
        """ output, attention = forward(inputs)
        inputs
        ------
          inputs: tensor, shape (batchsize, length, feature_dim)
        
        output
        ------
          output: tensor
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
          attention: tensor, shape (batchsize, length, num_head)
        """        
        # batch size
        batch_size = inputs.size(0)
        # feature dimension
        feat_dim = inputs.size(2)
        
        # input is (batch, legth, feature_dim)
        # change mm_weights to (batchsize, feature_dim, num_head)
        # weights will be in shape (batchsize, length, num_head)
        weights = torch.bmm(inputs, 
                            self.mm_weights.permute(1, 0).contiguous()\
                            .unsqueeze(0).repeat(batch_size, 1, 1))
        
        # attention (batchsize, length, num_head)
        attentions = torch_nn_func.softmax(torch.tanh(weights),dim=1)        
        
        # apply attention weight to input vectors
        if self.num_head == 1:
            # We can use the mode below to compute self.num_head too
            # But there is numerical difference.
            #  original implementation in github
            
            # elmentwise multiplication
            # weighted input vector: (batchsize, length, feature_dim)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            # weights_mat = (batch * length, feat_dim, num_head)
            weighted = torch.bmm(
                inputs.view(-1, feat_dim, 1), 
                attentions.view(-1, 1, self.num_head))
            
            # weights_mat = (batch, length, feat_dim * num_head)
            weighted = weighted.view(batch_size, -1, feat_dim * self.num_head)
            
        # pooling
        if self.mean_only:
            # only output the mean vector
            representations = weighted.sum(1)
        else:
            # output the mean and std vector
            noise = self.noise_std * torch.randn(
                weighted.size(), dtype=weighted.dtype, device=weighted.device)

            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)
            # concatenate mean and std
            representations = torch.cat((avg_repr,std_repr),1)
        # done
        return representations, attentions
    
    def forward(self, inputs):
        """ output = forward(inputs)
        inputs
        ------
          inputs: tensor, shape (batchsize, length, feature_dim)
        
        output
        ------
          output: tensor
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
        """
        output, _ = self._forward(inputs)
        return output

    def debug(self, inputs):
        return self._forward(inputs)


class Conv1dForARModel(Conv1dKeepLength):
    """ Definition of dilated Convolution for autoregressive model

    This module is based on block_nn.py/Conv1DKeepLength.
    However, Conv1DKeepLength doesn't assume step-by-step generation
    for autogressive model.

    This Module further adds the method to generate output in AR model

    Example:

        import torch
        import torch.nn as torch_nn
        import torch.nn.functional as torch_nn_func
        import sandbox.block_nn as nii_nn
        import tutorials.plot_lib as nii_plot

        # Compare the results of two layers
        batchsize = 1
        input_dim = 1
        output_dim = 1
        length = 5
        dilation = 2
        kernel_s = 3

        # Layers
        conv1 = nii_nn.Conv1dKeepLength(
               input_dim, output_dim, dilation, kernel_s,
               causal=True, tanh=False, bias=True)
        conv2 = Conv1dForARModel(input_dim, output_dim, dilation, kernel_s,
                                 tanh=False, bias=True)
        conv2.weight = conv1.weight
        conv2.bias = conv1.bias

        # Test
        input = torch.rand([batchsize, length, input_dim])
        with torch.no_grad():
            output = conv1(input)
            output2 = conv2(input)

            out = torch.zeros([batchsize, length, output_dim])
            for step in range(length):
                out[:, step:step+1, :] = conv2(input[:, step:step+1, :], step)

        print(output - output2)
        print(output - out)
        #nii_plot.plot_tensor(input, deci_width=2)
        #nii_plot.plot_tensor(output, deci_width=2)
        #nii_plot.plot_tensor(output2, deci_width=2)
        #nii_plot.plot_tensor(out, deci_width=2)
    """
    def __init__(self, input_dim, output_dim, dilation_s, kernel_s,
                 bias=True, tanh = True, causal=True):
        """ Conv1dForARModel(input_dim, output_dim, dilation_s, kernel_s,
            bias=True, tanh=True)

        args
        ----
          input_dim: int, input tensor should be (batchsize, length, input_dim)
          output_dim: int, output tensor will be (batchsize, length, output_dim)
          dilation_s: int, dilation size
          kernel_s: int, kernel size
          bias: bool, whether use bias term, default True
          tanh: bool, whether apply tanh on the output, default True
          causal: bool, whether the convoltuion is causal, default True
        
        Note that causal==False, step-by-step AR generation will raise Error
        """
        super(Conv1dForARModel, self).__init__(
            input_dim, output_dim, dilation_s, kernel_s, \
            causal = causal, stride = 1, groups=1, bias=bias, tanh = tanh)

        # configuration options
        self.use_bias = bias
        self.use_tanh = tanh
        self.kernel_s = kernel_s
        self.dilation_s = dilation_s
        self.out_dim = output_dim
        self.causal = causal

        # See slide http://tonywangx.github.io/slide.html#misc CURRENNT WaveNet,
        # page 50-56 for example on kernel_s = 2
        #
        # buffer length, depends on kernel size and dilation size
        # kernel_size = 3, dilation_size = 1 -> * * * -> buffer_len = 3
        # kernel_size = 3, dilation_size = 2 -> * . * . * -> buffer_len = 5
        self.buffer_len = (kernel_s - 1) * dilation_s + 1
        self.buffer_data = None
        # self.buffer_conv1d = None
        return

    def forward(self, x, step_idx = None):
        """ output = forward(x, step_idx)

        input
        -----
          x: tensor, in shape (batchsize, length, input_dim)
          step_idx: int, the index of the current time step
                    or None
        output
        ------
          output: tensor, in shape (batchsize, length, output_dim)

        If step_idx is True
        ------------------------
          this is same as common conv1d forward method

        If self.training is False
        ------------------------
          This method assumes input and output tensors
          are for one time step, i.e., length = 1 for both x and output.
          This method should be used in a loop, for example:

          model.eval()
          for idx in range(total_time_steps):
              ...
              output[:, idx:idx+1, :] = forward(x[:, idx:idx+1, :])
              ...

          This Module will use a buffer to store the intermediate results.
          See slide http://tonywangx.github.io/slide.html#misc CURRENNT WaveNet,
           page 50-56 for example on kernel_s = 2
        """
        if step_idx is None:
            # normal training mode, use the common conv forward method
            return super(Conv1dForARModel, self).forward(x)
        else:
            if self.causal is False:
                print("Step-by-step generation cannot work on non-causal conv")
                print("Please use causal=True for Conv1dForARModel")
                sys.exit(1)
            # step-by-step for generation in AR model

            # initialize buffer if necessary
            if step_idx == 0:
                self.buffer_data = torch.zeros(
                    [x.shape[0], self.buffer_len, x.shape[-1]],
                    dtype=x.dtype, device=x.device)
                #self.buffer_conv1d = torch.zeros(
                #    [x.shape[0], self.kernel_s, x.shape[-1]],
                #    dtype=x.dtype, device=x.device)

            # Put new input data into buffer
            #  the position index to put the input data
            tmp_ptr_save = step_idx % self.buffer_len
            #  assume x is (batchsize, length=1, input_dim), thus
            #  only take x[:, 0, :]
            self.buffer_data[:, tmp_ptr_save, :] = x[:, 0, :]
            
            ## Method 1: do multiplication and summing
            ## 
            ##  initialize
            #output_tensor = torch.zeros(
            #    [x.shape[0], self.out_dim], dtype=x.dtype, device=x.device)
            ##  loop over the kernel
            #for ker_idx in range(self.kernel_s):
            #    # which buffer should be retrieved for this kernel idx
            #    tmp_data_idx = (step_idx - ker_idx * self.dilation_s) \
            #                   % self.buffer_len
            #    # apply the kernel and sum the product
            #    # note that self.weight[:, :, -1] is the 1st kernel
            #    output_tensor += torch.matmul(
            #        self.buffer_data[:, tmp_data_idx, :],
            #        self.weight[:, :, self.kernel_s - ker_idx - 1].T)

            ## Method 2: take advantage of conv1d API
            # Method 2 is slower than Method1 when kernel size is small
            ## create a input buffer to conv1d
            #idxs = [(step_idx - x * self.dilation_s) % self.buffer_len \
            #        for x in range(self.kernel_s)][::-1]
            #self.buffer_conv1d = self.buffer_data[:, idxs, :].permute(0, 2, 1)
            #output_tensor = torch_nn_func.conv1d(self.buffer_conv1d, 
            #                                     self.weight).permute(0, 2, 1)
            
            # Method 3:
            batchsize = x.shape[0]
            # which data buffer should be retrieved for each kernel
            #  [::-1] is necessary because self.weight[:, :, -1] corresponds to
            #  the first buffer, [:, :, -2] to the second ...
            index_buf = [(step_idx - y * self.dilation_s) % self.buffer_len \
                         for y in range(self.kernel_s)][::-1]
            # concanate buffers as a tensor [batchsize, input_dim * kernel_s]
            # concanate weights as a tensor [input_dim * kernel_s, output_dim]
            # (out_dim, in_dim, kernel_s)-permute->(out_dim, kernel_s, in_dim)
            # (out_dim, kernel_s, in_dim)-reshape->(out_dim, in_dim * kernel_s)
            output_tensor = torch.mm(
                self.buffer_data[:, index_buf, :].view(batchsize, -1),
                self.weight.permute(0, 2, 1).reshape(self.out_dim, -1).T)


            # apply bias and tanh if necessary
            if self.use_bias:
                output_tensor += self.bias
            if self.use_tanh:
                output_tensor = torch.tanh(output_tensor)

            # make it to (batch, length=1, output_dim)
            return output_tensor.unsqueeze(1)


class AdjustTemporalResoIO(torch_nn.Module):
    def __init__(self, list_reso, target_reso, list_dims):
        """AdjustTemporalResoIO(list_reso, target_reso, list_dims)
        Module to change temporal resolution of input tensors.        

        Args
        ----
          list_reso: list, list of temporal resolutions. 
            list_reso[i] should be the temporal resolution of the 
            (i+1)-th tensor 
          target_reso: int, target temporal resolution to be changed
          list_dims: list, list of feat_dim for tensors
             assume tensor to have shape (batchsize, time length, feat_dim)

        Note 
        ----
          target_reso must be <= max(list_reso)
          all([target_reso % x == 0 for x in list_reso if x < target_reso]) 
          all([x % target_reso == 0 for x in list_reso if x < target_reso])

        Suppose a tensor A (batchsize, time_length1, feat_dim_1) has 
        temporal resolution of 1. Tensor B has temporal resolution
        k and is aligned with A. Then B[:, n, :] corresponds to
        A[:, k*n:k*(n+1), :].
        
        For example: 
        let k = 3, batchsize = 1, feat_dim = 1
        ---------------> time axis
                         0 1 2 3 4 5 6 7 8
        A[0, 0:9, 0] = [ a b c d e f g h i ]
        B[0, 0:3, 0] = [ *     &     ^     ]
        
        [*] is aligned with [a b c]
        [&] is aligned with [d e f]
        [^] is aligned with [g h i]

        Assume the input tensor list is [A, B]:
          list_reso = [1, 3]
          list_dims = [A.shape[-1], B.shape[-1]]
        
        If target_reso = 3, then
          B will not be changed
          A (batchsize=1, time_length1=9, feat_dim=1) will be A_new (1, 3, 3)
          
          B    [0, 0:3, 0] = [  *     &     ^     ]
          A_new[0, 0:3, :] = [ [a,   [d,   [g,    ]
                                b,    e,    h,
                                c]    f]    i]
        
        More concrete examples:
        input_dims = [5, 3]
        rates = [1, 6]
        target_rate = 2
        l_adjust = AdjustTemporalRateIO(rates, target_rate, input_dims)
        data1 = torch.rand([2, 2*6, 5])
        data2 = torch.rand([2, 2, 3])
        data1_new, data2_new = l_adjust([data1, data2])
        
        # Visualization requires matplotlib and tutorial.plot_lib as nii_plot
        nii_plot.plot_tensor(data1)
        nii_plot.plot_tensor(data1_new)

        nii_plot.plot_tensor(data2)
        nii_plot.plot_tensor(data2_new)
        """
        
        super(AdjustTemporalResoIO, self).__init__()
        list_reso = np.array(list_reso)
        list_dims = np.array(list_dims)

        # save
        self.list_reso = list_reso
        self.fatest_reso = min(list_reso)
        self.slowest_reso = max(list_reso)
        self.target_reso = target_reso
        
        # check
        if any(list_reso < 0):
            print("Expects positive resolution in AdjustTemporalResoIO")
            sys.exit(1)

        if self.target_reso < 0:
            print("Expects positive target_reso in AdjustTemporalResoIO")
            sys.exit(1)

        if any([x % self.target_reso != 0 for x in self.list_reso \
                if x > self.target_reso]):
            print("Resolution " + str(list_reso) + " incompatible")
            print(" with target resolution {:d}".format(self.target_reso))
            sys.exit(1)

        if any([self.target_reso % x != 0 for x in self.list_reso \
                if x < self.target_reso]):
            print("Resolution " + str(list_reso) + " incompatible")
            print(" with target resolution {:d}".format(self.target_reso))
            sys.exit(1)
        
        self.dim_change = []
        self.reso_change = []
        self.l_upsampler = []
        for x, dim in zip(self.list_reso, list_dims):
            if x > self.target_reso:
                # up sample
                # up-sample don't change feat dim, just duplicate frames
                self.dim_change.append(1)
                self.reso_change.append(x // self.target_reso)
                self.l_upsampler.append(
                    nii_nn.UpSampleLayer(dim, x // self.target_reso))
            elif x < self.target_reso:
                # down sample
                # for down-sample, we fold the multiple feature frames into one 
                self.dim_change.append(self.target_reso // x)
                # use a negative number to indicate down-sample
                self.reso_change.append(-self.target_reso // x)
                self.l_upsampler.append(None)
            else:
                self.dim_change.append(1)
                self.reso_change.append(1)
                self.l_upsampler.append(None)
                
        self.l_upsampler = torch_nn.ModuleList(self.l_upsampler)
        
        # log down the dimensions after resolution change
        self.dim = []
        if list_dims is not None and len(list_dims) == len(self.dim_change):
            self.dim = [x * y for x, y in zip(self.dim_change, list_dims)]
        

        return
    
    def get_dims(self):
        return self.dim
    
    
    def forward(self, tensor_list):
        """ tensor_list = AdjustTemporalResoIO(tensor_list):
        Adjust the temporal resolution of the input tensors.
        For up-sampling, the tensor is duplicated
        For down-samplin, multiple time steps are concated into a single vector
        
        input
        -----
          tensor_list: list, list of tensors,  
                       (batchsize, time steps, feat dim)
          
        output
        ------
          tensor_list: list, list of tensors, 
                       (batchsize, time_steps * N, feat_dim * Y)
        
        where N is the resolution change option in self.reso_change, 
        Y is the factor to change dimension in self.dim_change
        """

        output_tensor_list = []
        for in_tensor, dim_fac, reso_fac, l_up in \
            zip(tensor_list, self.dim_change, self.reso_change, 
                self.l_upsampler):
            batchsize = in_tensor.shape[0]
            timelength = in_tensor.shape[1]
            
            if reso_fac == 1:
                # no change
                output_tensor_list.append(in_tensor)
            elif reso_fac < 0:
                # down sample by concatenating
                reso_fac *= -1
                expected_len = timelength // reso_fac
                trim_length = expected_len * reso_fac
                
                if expected_len == 0:
                    # if input tensor length < down_sample factor
                    output_tensor_list.append(
                        torch.reshape(in_tensor[:, 0:1, :], 
                                      (batchsize, 1, -1)))
                else:
                    # make sure that 
                    output_tensor_list.append(
                        torch.reshape(in_tensor[:, 0:trim_length, :], 
                                      (batchsize, expected_len, -1)))
            else:
                # up-sampling by duplicating
                output_tensor_list.append(l_up(in_tensor))
        return output_tensor_list

class LSTMZoneOut(torch_nn.Module):
    """LSTM layer with zoneout
    This module replies on LSTMCell
    """
    def __init__(self, in_feat_dim, out_feat_dim, 
                 bidirectional=False, residual_link=False, bias=True):
        """LSTMZoneOut(in_feat_dim, out_feat_dim, 
                       bidirectional=False, residual_link=False, bias=True)
        
        Args
        ----
          in_feat_dim: int, input tensor should be (batch, length, in_feat_dim)
          out_feat_dim: int, output tensor will be (batch, length, out_feat_dim)
          bidirectional: bool, whether bidirectional, default False
          residual_link: bool, whether residual link over LSTM, default False
          bias: bool, bias option in torch.nn.LSTMCell, default True
          
        When bidirectional is True, out_feat_dim must be an even number
        When residual_link is True, out_feat_dim must be equal to in_feat_dim
        """
        super(LSTMZoneOut, self).__init__()
        
        # config parameters
        self.in_dim = in_feat_dim
        self.out_dim = out_feat_dim
        self.flag_bi = bidirectional
        self.flag_res = residual_link
        self.bias = bias
        
        # check
        if self.flag_res and self.out_dim != self.in_dim:
            print("Error in LSTMZoneOut w/ residual: in_feat_dim!=out_feat_dim")
            sys.exit(1)
            
        if self.flag_bi and self.out_dim % 2 > 0:
            print("Error in Bidirecional LSTMZoneOut: out_feat_dim is not even")
            sys.exit(1)
        
        # layer
        if self.flag_bi:
            self.l_lstm1 = torch_nn.LSTMCell(
                self.in_dim, self.out_dim//2, self.bias)
            self.l_lstm2 = torch_nn.LSTMCell(
                self.in_dim, self.out_dim//2, self.bias)
        else:
            self.l_lstm1 = torch_nn.LSTMCell(
                self.in_dim, self.out_dim, self.bias)
            self.l_lstm2 = None
        return
    
    def _zoneout(self, pre, cur, p=0.1):
        """zoneout wrapper
        """
        if self.training:
            with torch.no_grad():
                mask = torch.zeros_like(pre).bernoulli_(p)
            return pre * mask + cur * (1-mask)
        else:
            return cur
    
    def forward(self, x):
        """y = LSTMZoneOut(x)
        
        input
        -----
          x: tensor, (batchsize, length, in_feat_dim)
          
        output
        ------
          y: tensor, (batchsize, length, out_feat_dim)
        """
        batchsize = x.shape[0]
        length = x.shape[1]
        
        # output tensor
        y = torch.zeros([batchsize, length, self.out_dim], 
                        device=x.device, dtype=x.dtype)
            
        # recurrent 
        if self.flag_bi:
            # for bi-directional 
            hid1 = torch.zeros([batchsize, self.out_dim//2], 
                               device=x.device, dtype=x.dtype) 
            hid2 = torch.zeros_like(hid1)
            cell1 = torch.zeros_like(hid1)
            cell2 = torch.zeros_like(hid1)
            
            for time in range(length):
                # reverse time idx
                rtime = length-time-1
                # compute in both forward and reverse directions
                hid1_new, cell1_new = self.l_lstm1(x[:,time, :], (hid1, cell1))
                hid2_new, cell2_new = self.l_lstm2(x[:,rtime, :], (hid2, cell2))
                hid1 = self._zoneout(hid1, hid1_new)
                hid2 = self._zoneout(hid2, hid2_new)
                y[:, time, 0:self.out_dim//2] = hid1
                y[:, length-time-1, self.out_dim//2:] = hid2
        
        else:
            # for uni-directional    
            hid1 = torch.zeros([batchsize, self.out_dim], 
                               device=x.device, dtype=x.dtype) 
            cell1 = torch.zeros_like(hid1)
            
            for time in range(length):
                hid1_new, cell1_new = self.l_lstm1(x[:, time, :], (hid1, cell1))
                hid1 = self._zoneout(hid1, hid1_new)
                y[:, time, :] = hid1

        # residual part
        if self.flag_res:
            y = y+x
        return y
        

class LinearInitialized(torch_nn.Module):
    """Linear layer with specific initialization
    """
    def __init__(self, weight_mat, flag_train=True):
        """LinearInitialized(weight_mat, flag_trainable=True)
        
        Args
        ----
          weight_mat: tensor, (input_dim, output_dim), 
             the weight matrix for initializing the layer
          flag_train: bool, where trainable or fixed, default True
          
        This can be used for trainable filter bank. For example:
        import sandbox.util_frontend as nii_front_end
        l_fb = LinearInitialized(nii_front_end.linear_fb(fn, sr, filter_num))
        y = l_fb(x)
        """
        super(LinearInitialized, self).__init__()
        self.weight = torch_nn.Parameter(weight_mat, requires_grad=flag_train)
        return
    
    def forward(self, x):
        """y = LinearInitialized(x)
        
        input
        -----
          x: tensor, (batchsize, ..., input_feat_dim)
          
        output
        ------
          y: tensor, (batchsize, ..., output_feat_dim)
          
        Note that weight is in shape (input_feat_dim, output_feat_dim)
        """
        return torch.matmul(x, self.weight)



class GRULayer(torch_nn.Module):
    """GRULayer

    There are two modes for forward
    1. forward(x) -> process sequence x
    2. forward(x[n], n) -> process n-th step of x
    
    Example:
        data = torch.randn([2, 10, 3])
        m_layer = GRULayer(3, 3)
        out = m_layer(data)

        out_2 = torch.zeros_like(out)
        for idx in range(data.shape[1]):
            out_2[:, idx:idx+1, :] = m_layer._forwardstep(
               data[:, idx:idx+1, :], idx)
    """
    
    def __init__(self, in_size, out_size, flag_bidirec=False):
        """GRULayer(in_size, out_size, flag_bidirec=False)
        
        Args
        ----
          in_size: int, dimension of input feature per step
          out_size: int, dimension of output feature per step
          flag_bidirec: bool, whether this is bi-directional GRU
        """
        super(GRULayer, self).__init__()
        
        self.m_in_size = in_size
        self.m_out_size = out_size
        self.m_flag_bidirec = flag_bidirec
        
        self.m_gru = torch_nn.GRU(
            in_size, out_size, batch_first=True, 
            bidirectional=flag_bidirec)
            
        # for step-by-step generation
        self.m_grucell = None
        self.m_buffer = None
        return 
    
    def _get_gru_cell(self):
        # dump GRU layer to GRU cell for step-by-step generation
        self.m_grucell = torch_nn.GRUCell(self.m_in_size, self.m_out_size)
        self.m_grucell.weight_hh.data = self.m_gru.weight_hh_l0.data
        self.m_grucell.weight_ih.data = self.m_gru.weight_ih_l0.data
        self.m_grucell.bias_hh.data = self.m_gru.bias_hh_l0.data
        self.m_grucell.bias_ih.data = self.m_gru.bias_ih_l0.data
        return

    def _forward(self, x):
        """y = _forward(x)
        input
        -----
          x: tensor, (batch, length, inputdim)
          
        output
        ------
          y: tensor, (batch, length, out-dim)
        """
        out, hn = self.m_gru(x)
        return out
    
    def _forwardstep(self, x, step_idx):
        """y = _forwardstep(x)
        input
        -----
          x: tensor, (batch, 1, inputdim)
          
        output
        ------
          y: tensor, (batch, 1, out-dim)
        """
        if self.m_flag_bidirec:
            print("Bi-directional GRU not supported for step-by-step mode")
            sys.exit(1)
        else:
            if step_idx == 0:
                # load weight as grucell
                if self.m_grucell is None:
                    self._get_gru_cell()
                # buffer 
                self.m_buffer = torch.zeros(
                    [x.shape[0], self.m_out_size], 
                    device=x.device, dtype=x.dtype)
            self.m_buffer = self.m_grucell(x[:, 0, :], self.m_buffer)
            # (batch, dim) -> (batch, 1, dim)
            return self.m_buffer.unsqueeze(1)
            
    def forward(self, x, step_idx=None):
        """y = forward(x, step_idx=None)
        input
        -----
          x: tensor, (batch, length, inputdim)
          
        output
        ------
          y: tensor, (batch, length, out-dim)
          
        When step_idx >= 0, length must be 1, forward(x[:, n:n+1, :], n) 
        will process the x at the n-th step. The hidden state will be saved
        in the buffer and used for n+1 step
        """
        if step_idx is None:
            # given full context
            return self._forward(x)
        else:
            # step-by-step processing
            return self._forwardstep(x, step_idx)

class LSTMLayer(torch_nn.Module):
    """LSTMLayer

    There are two modes for forward
    1. forward(x) -> process sequence x
    2. forward(x[n], n) -> process n-th step of x
    
    Example:
        data = torch.randn([2, 10, 3])
        m_layer = LSTMLayer(3, 3)
        out = m_layer(data)

        out_2 = torch.zeros_like(out)
        for idx in range(data.shape[1]):
            out_2[:, idx:idx+1, :] = m_layer._forwardstep(
               data[:, idx:idx+1, :], idx)
    """
    
    def __init__(self, in_size, out_size, flag_bidirec=False):
        """LSTMLayer(in_size, out_size, flag_bidirec=False)
        
        Args
        ----
          in_size: int, dimension of input feature per step
          out_size: int, dimension of output feature per step
          flag_bidirec: bool, whether this is bi-directional GRU
        """
        super(LSTMLayer, self).__init__()
        
        self.m_in_size = in_size
        self.m_out_size = out_size
        self.m_flag_bidirec = flag_bidirec
        
        self.m_lstm = torch_nn.LSTM(
            input_size=in_size, hidden_size=out_size, 
            batch_first=True, 
            bidirectional=flag_bidirec)
            
        # for step-by-step generation
        self.m_lstmcell = None
        self.m_c_buf = None
        self.m_h_buf = None
        return 
    
    def _get_lstm_cell(self):
        # dump LSTM layer to LSTM cell for step-by-step generation
        self.m_lstmcell = torch_nn.LSTMCell(self.m_in_size, self.m_out_size)
        self.m_lstmcell.weight_hh.data = self.m_lstm.weight_hh_l0.data
        self.m_lstmcell.weight_ih.data = self.m_lstm.weight_ih_l0.data
        self.m_lstmcell.bias_hh.data = self.m_lstm.bias_hh_l0.data
        self.m_lstmcell.bias_ih.data = self.m_lstm.bias_ih_l0.data
        return

    def _forward(self, x):
        """y = _forward(x)
        input
        -----
          x: tensor, (batch, length, inputdim)
          
        output
        ------
          y: tensor, (batch, length, out-dim)
        """
        out, hn = self.m_lstm(x)
        return out
    
    def _forwardstep(self, x, step_idx):
        """y = _forwardstep(x)
        input
        -----
          x: tensor, (batch, 1, inputdim)
          
        output
        ------
          y: tensor, (batch, 1, out-dim)
        """
        if self.m_flag_bidirec:
            print("Bi-directional GRU not supported for step-by-step mode")
            sys.exit(1)
        else:
            
            if step_idx == 0:
                # For the 1st time step, prepare the LSTM Cell and buffer
                # load weight as LSTMCell
                if self.m_lstmcell is None:
                    self._get_lstm_cell()
                    
                # buffer 
                self.m_c_buf = torch.zeros([x.shape[0], self.m_out_size], 
                                           device=x.device, dtype=x.dtype)
                self.m_h_buf = torch.zeros_like(self.m_c_buf)
                
            # do generation
            self.m_h_buf, self.m_c_buf = self.m_lstmcell(
                x[:, 0, :], (self.m_h_buf, self.m_c_buf))
            
            # (batch, dim) -> (batch, 1, dim)
            return self.m_h_buf.unsqueeze(1)
            
    def forward(self, x, step_idx=None):
        """y = forward(x, step_idx=None)
        input
        -----
          x: tensor, (batch, length, inputdim)
          
        output
        ------
          y: tensor, (batch, length, out-dim)
          
        When step_idx >= 0, length must be 1, forward(x[:, n:n+1, :], n) 
        will process the x at the n-th step. The hidden state will be saved
        in the buffer and used for n+1 step
        """
        if step_idx is None:
            # given full context
            return self._forward(x)
        else:
            # step-by-step processing
            return self._forwardstep(x, step_idx)

class DropoutForMC(torch_nn.Module):
    """Dropout layer for Bayesian model
    THe difference is that we do dropout even in eval stage
    """
    def __init__(self, p, dropout_flag=True):
        super(DropoutForMC, self).__init__()
        self.p = p
        self.flag = dropout_flag
        return
        
    def forward(self, x):
        return torch_nn_func.dropout(x, self.p, training=self.flag)
        

if __name__ == "__main__":
    print("Definition of block NN")
