import numpy as np

# -------------------------------
# Flattening Layer
# -------------------------------

class Concatenation:
    '''
    Flattening layer.
    '''

    def __init__(self, layer):
        self.concat_axis = layer.axis
        self.last_index = 0

    def lrp(self, R, X):
        '''
        Receives upper layer input relevance R and reshapes it to match the input neurons.
        '''
        # just propagate R further down.
        # makes sure subroutines never get called.
        Rx = np.take(R, [i for i in range(self.last_index, self.last_index + X.shape[self.concat_axis])], axis=self.concat_axis)
        self.last_index += X.shape[self.concat_axis]
        return Rx