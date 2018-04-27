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
