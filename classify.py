import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv, os
import math
from PIL import Image

from keras.models import load_model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from tensorflow.python.keras.optimizers import Adam

img_size = 512

img_size_flat = img_size * img_size * 3

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_size, img_size, 3)

class_name = "neck_design_labels"
model_file = "Xception0.9/neck_design_labels_model.h5"
weight_file = "Xception0.9/weights.hdf5"
model = load_model(os.path.join("models", class_name, model_file))
model.load_weights(os.path.join("models", class_name, weight_file))

tests = []
rows = []
index = 0
with open('rank/Tests/question.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[1] != class_name:
            continue
        image = Image.open("rank/" + row[0])
        img_array = np.asarray(image)
        if img_array.shape != img_shape_full:
            image = image.resize((img_size, img_size), Image.ANTIALIAS)
            img_array = np.asarray(image)
        tests.append(img_array)
        rows.append(row)
        index += 1
        if index % 500 == 0:
            print(index)
results = model.predict(np.array(tests), batch_size=16, verbose=0, steps=None)

with open('results.csv', 'w') as wfile:
    writer = csv.writer(wfile, delimiter=',')

    for index in range(len(rows)):
        result = results[index]
        ans = ""
        for r in result:
            ans += str(int(r * 10000) / 10000)
            ans += ";"
        ans = ans[:-1]
        
        row = rows[index]
        row[-1] = ans
        print(row)
        writer.writerow(row)
