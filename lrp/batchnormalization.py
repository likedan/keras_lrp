import numpy as np
import tensorflow as tf
na = np.newaxis

class BatchNormalization:

    def __init__(self, layer, X):

        self.X = X
        self.W = layer.get_weights()[0]

    def _simple_lrp(self, R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        Rx = R * self.W

        return Rx
