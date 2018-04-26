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
import tensorflow as tf
na = np.newaxis


# -------------------------------
# 2D Convolution layer
# -------------------------------

class Convolution:

    def __init__(self, layer, X):

        self.X = X

        self.fh, self.fw = layer.kernel_size
        self.n = layer.filters

        self.W = layer.get_weights()[0]
        if len(layer.get_weights()) == 1:
            self.B = np.zeros([self.n])
        else:
            self.B = layer.get_weights()[1]
        self.stride = layer.strides
        self.padding = layer.padding

    def _simple_lrp(self, R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        if self.padding == "valid":
            Rx = np.zeros_like(self.X, dtype=np.float)
            print(self.stride, R.shape, self.X.shape)
            for i in range(Hout):
                for j in range(Wout):
                    Z = self.W[na,...] * self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : , na]
                    Zs = Z.sum(axis=(1,2,3),keepdims=True) + self.B[na,na,na,na,...]
                    Zs += 1e-12*((Zs >= 0)*2 - 1.) # add a weak numerical stabilizer to cushion division by zero
                    Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
            return Rx
        else:
            shape = (self.X.shape[0], self.X.shape[1] + hf - 1, self.X.shape[2] + wf - 1, self.X.shape[3])
            Rx = np.zeros(shape)
            new_X = np.zeros(shape)
            print(self.stride, new_X.shape, Rx.shape, self.X.shape, R.shape)
            new_X[:, int(hstride/2):Hout+int(hstride/2) , int(wstride/2):Wout+int(wstride/2), :] = self.X
            self.X = new_X
            for i in range(Hout):
                for j in range(Wout):
                    Z = self.W[na,...] * self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : , na]
                    Zs = Z.sum(axis=(1,2,3),keepdims=True) + self.B[na,na,na,na,...]
                    Zs += 1e-12*((Zs >= 0)*2 - 1.) # add a weak numerical stabilizer to cushion division by zero
                    Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
            return Rx[:, int(hstride/2):Hout+int(hstride/2) , int(wstride/2):Wout+int(wstride/2), :]

    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = np.zeros_like(self.X,dtype=np.float)

        for i in range(Hout):
            for j in range(Wout):
                Z = np.ones((N,hf,wf,df,NF))
                Zs = Z.sum(axis=(1,2,3),keepdims=True)

                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
        return Rx

    def _ww_lrp(self,R):
        '''
        LRP according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = np.zeros_like(self.X,dtype=np.float)

        for i in range(Hout):
            for j in range(Wout):
                Z = self.W[na,...]**2
                Zs = Z.sum(axis=(1,2,3),keepdims=True)

                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
        return Rx

    def _epsilon_lrp(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = np.zeros_like(self.X,dtype=np.float)

        for i in range(Hout):
            for j in range(Wout):
                Z = self.W[na,...] * self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : , na]
                Zs = Z.sum(axis=(1,2,3),keepdims=True) + self.B[na,na,na,na,...]
                Zs += epsilon*((Zs >= 0)*2-1)
                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
        return Rx


    def _alphabeta_lrp(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''

        beta = 1 - alpha

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = np.zeros_like(self.X,dtype=np.float)

        for i in range(Hout):
            for j in range(Wout):
                Z = self.W[na,...] * self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : , na]

                if not alpha == 0:
                    Zp = Z * (Z > 0)
                    Bp = (self.B * (self.B > 0))[na,na,na,na,...]
                    Zsp = Zp.sum(axis=(1,2,3),keepdims=True) + Bp
                    Ralpha = alpha * ((Zp/Zsp) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
                else:
                    Ralpha = 0

                if not beta == 0:
                    Zn = Z * (Z < 0)
                    Bn = (self.B * (self.B < 0))[na,na,na,na,...]
                    Zsn = Zn.sum(axis=(1,2,3),keepdims=True) + Bn
                    Rbeta = beta * ((Zn/Zsn) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
                else:
                    Rbeta = 0

                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += Ralpha + Rbeta

        return Rx
