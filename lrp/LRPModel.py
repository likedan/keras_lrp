import tensorflow as tf
from lrp.convolution import Convolution
from lrp.linear import Linear
from lrp.avgpool import AvgPool
from lrp.maxpool import MaxPool
from lrp.globalavgpool import GlobalAvgPool
from lrp.concatenation import Concatenation
from lrp.batchnormalization import BatchNormalization
import numpy as np

class LRPModel(tf.keras.Model):

    def __init__(self, model):

        self.source_model = model
        all_outputs = []
        for layer in model.layers:
            all_outputs.append(layer.output)
        tf.keras.Model.__init__(self, model.inputs, all_outputs)

        print(model.summary())

    def perform_lrp(self, X):
        output = self.predict(X)
        layer_R = {}
        layer_R[self.source_model.layers[-1].name] = output[-1]
        for layer in reversed(self.source_model.layers):

            from_layers = layer._inbound_nodes[0].inbound_layers

            if type(layer) is tf.keras.layers.Concatenate:
                r_layer = Concatenation(layer)
                for from_layer in from_layers:
                    print(layer.name, from_layer.name)
                    from_layer_index = self.source_model.layers.index(from_layer)
                    # print(layer_R[layer.name].shape, output[from_layer_index].shape)
                    layer_R[from_layer.name] = r_layer.lrp(layer_R[layer.name], output[from_layer_index])

            elif len(from_layers) == 1:
                from_layer = from_layers[0]
                from_layer_index = self.source_model.layers.index(from_layer)
                layer_index = self.source_model.layers.index(layer)
                if type(layer) is tf.keras.layers.Conv2D:
                    r_layer = Convolution(layer, output[from_layer_index])
                elif type(layer) is tf.keras.layers.Dense:
                    r_layer = Linear(layer, output[from_layer_index])
                elif type(layer) is tf.keras.layers.AveragePooling2D:
                    r_layer = AvgPool(layer, output[from_layer_index])
                elif type(layer) is tf.keras.layers.MaxPooling2D:
                    r_layer = MaxPool(layer, output[from_layer_index], output[layer_index])
                elif type(layer) is tf.keras.layers.GlobalAveragePooling2D:
                    r_layer = GlobalAvgPool(layer, output[from_layer_index])
                elif type(layer) is tf.keras.layers.BatchNormalization:
                    r_layer = BatchNormalization(layer, output[from_layer_index])
                else:
                    print("skip", layer.name)
                    layer_R[from_layer.name] = layer_R[layer.name]
                    continue
                layer_R[from_layer.name] = r_layer._simple_lrp(layer_R[layer.name])
            else:
                #done
                break

            print(layer.name, from_layer.name, np.average(layer_R[layer.name]))

        return layer_R[self.source_model.layers[0].name]