'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 20.10.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import numpy as np

# -------------------------------
# Max Pooling layer
# -------------------------------

class MaxPool:

    def __init__(self, layer, X, Y):

        self.X = X
        self.Y = Y
        self.pool = layer.pool_size
        self.stride = layer.strides


    # def forward(self,X):
    #     '''
    #     Realizes the forward pass of an input through the max pooling layer.
    #
    #     Parameters
    #     ----------
    #     X : numpy.ndarray
    #         a network input, shaped (N,H,W,D), with
    #         N = batch size
    #         H, W, D = input size in heigth, width, depth
    #
    #     Returns
    #     -------
    #     Y : numpy.ndarray
    #         the max-pooled outputs, reduced in size due to given stride and pooling size
    #     '''
    #
    #     self.X = X
    #     N,H,W,D = X.shape
    #
    #     hpool,   wpool   = self.pool
    #     hstride, wstride = self.stride
    #
    #     #assume the given pooling and stride parameters are carefully chosen.
    #     Hout = (H - hpool) / hstride + 1
    #     Wout = (W - wpool) / wstride + 1
    #
    #     #initialize pooled output
    #     self.Y = np.zeros((N,Hout,Wout,D))
    #
    #     for i in range(Hout):
    #         for j in range(Wout):
    #             self.Y[:,i,j,:] = X[:, i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ].max(axis=(1,2))
    #     return self.Y
    #




    def _simple_lrp(self,R):
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = int((H - hpool) / hstride) + 1
        Wout = int((W - wpool) / wstride) + 1

        Rx = np.zeros_like(self.X,dtype=np.float)

        for i in range(Hout):
            for j in range(Wout):
                Z = self.Y[:,i:i+1,j:j+1,:] == self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ]
                Zs = Z.sum(axis=(1,2),keepdims=True,dtype=np.float) #thanks user wodtko for reporting this bug/fix
                Rx[:,i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool,:] += (Z / Zs) * R[:,i:i+1,j:j+1,:]
        return Rx


    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        '''
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        Rx = np.zeros_like(self.X,dtype=np.float)

        for i in range(Hout):
            for j in range(Wout):
                Z = np.ones([N,hpool,wpool,D])
                Zs = Z.sum(axis=(1,2),keepdims=True)
                Rx[:,i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool,:] += (Z / Zs) * R[:,i:i+1,j:j+1,:]
        return Rx

    def _ww_lrp(self,R):
        '''
        There are no weights to use. default to _flat_lrp(R)
        '''
        return self._flat_lrp(R)

    def _epsilon_lrp(self,R,epsilon):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R)

    def _alphabeta_lrp(self,R,alpha):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R)