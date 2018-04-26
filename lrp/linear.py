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
na = np.newaxis

# -------------------------------
# Linear layer
# -------------------------------
class Linear:
    '''
    Linear Layer
    '''

    def __init__(self, layer, X):

        self.X = X
        self.W = layer.get_weights()[0]
        if len(layer.get_weights()) == 1:
            self.B = np.zeros([layer.shape[1]])
        else:
            self.B = layer.get_weights()[1]


    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        Z = self.W[na,:,:]*self.X[:,:,na] #localized preactivations
        Zs = Z.sum(axis=1)[:,na,:] +self.B[na,na,:] #preactivations
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)

    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        note that for fully connected layers, this results in a uniform lower layer relevance map.
        '''
        Z = np.ones_like(self.W[na,:,:])
        Zs = Z.sum(axis=1)[:,na,:]
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)

    def _ww_lrp(self,R):
        '''
        LRR according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
        '''
        Z = self.W[na,:,:]**2
        Zs = Z.sum(axis=1)[:,na,:]
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)

    def _epsilon_lrp(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        Z = self.W[na,:,:]*self.X[:,:,na] # localized preactivations
        Zs = Z.sum(axis=1)[:,na,:] +self.B[na,na,:] # preactivations

        # add slack to denominator. we require sign(0) = 1. since np.sign(0) = 0 would defeat the purpose of the numeric stabilizer we do not use it.
        Zs += epsilon * ((Zs >= 0)*2-1)
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)


    def _alphabeta_lrp(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''
        beta = 1 - alpha
        Z = self.W[na,:,:]*self.X[:,:,na] # localized preactivations

        if not alpha == 0:
            Zp = Z * (Z > 0)
            Zsp = Zp.sum(axis=1)[:,na,:] + (self.B * (self.B > 0))[na,na,:]
            Ralpha = alpha * ((Zp / Zsp) * R[:,na,:]).sum(axis=2)
        else:
            Ralpha = 0

        if not beta == 0:
            Zn = Z * (Z < 0)
            Zsn = Zn.sum(axis=1)[:,na,:] + (self.B * (self.B < 0))[na,na,:]
            Rbeta = beta * ((Zn / Zsn) * R[:,na,:]).sum(axis=2)
        else:
            Rbeta = 0

        return Ralpha + Rbeta