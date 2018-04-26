'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import numpy as np

# -------------------------------
# Sum Pooling layer
# -------------------------------

class GlobalAvgPool:

    def __init__(self, layer, X):
        '''
        Constructor for the sum pooling layer object

        Parameters
        ----------

        pool : tuple (h,w)
            the size of the pooling mask in vertical (h) and horizontal (w) direction

        stride : tuple (h,w)
            the vertical (h) and horizontal (w) step sizes between filter applications.
        '''

        self.X = X

    def _simple_lrp(self, R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        N,H,W,D = self.X.shape

        Rx = np.zeros(self.X.shape)
        for i in range(D):
            Rx[0, :, :, i] = np.zeros((H, W)) + R[0, i]

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
        due to uniform weights used for sum pooling (1), this method defaults to _flat_lrp(R)
        '''
        return self._flat_lrp(R)

    def _epsilon_lrp(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        Rx = np.zeros(self.X.shape)
        for i in range(Hout):
            for j in range(Wout):
                Z = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations.
                Zs = Z.sum(axis=(1,2),keepdims=True)
                Zs += epsilon*((Zs >= 0)*2-1) # add a epsilon stabilizer to cushion an all-zero input

                Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += (Z/Zs) * R[:,i:i+1,j:j+1,:]  #distribute relevance propoprtional to input activations per layer

        return Rx


    # yes, we can do this. no, it will not make sense most of the time.  by default, _lrp_simple will be called. see line 152
    def _alphabeta_lrp(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''

        beta = 1-alpha

        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        #distribute the gradient towards across all inputs evenly
        Rx = np.zeros(self.X.shape)
        for i in range(Hout):
            for j in range(Wout):
                Z = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations.

                if not alpha == 0:
                    Zp = Z * (Z > 0)
                    Zsp = Zp.sum(axis=(1,2),keepdims=True) +1e-16 #zero division is quite likely in sum pooling layers when using the alpha-variant
                    Ralpha = (Zp/Zsp) * R[:,i:i+1,j:j+1,:]
                else:
                    Ralpha = 0

                if not beta == 0:
                    Zn = Z * (Z < 0)
                    Zsn = Zn.sum(axis=(1,2),keepdims=True) - 1e-16 #zero division is quite likely in sum pooling layers when using the alpha-variant
                    Rbeta = (Zn/Zsn) * R[:,i:i+1,j:j+1,:]
                else:
                    Rbeta = 0

                Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += Ralpha + Rbeta

        return Rx
