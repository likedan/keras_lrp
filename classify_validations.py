# go over the models dir and classify everything

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv, os
import math
from PIL import Image

from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.models import load_model
from tensorflow.python.keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from vis.utils import utils
from keras import activations
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay
from keras.layers import Dense, SeparableConv2D
from keras.optimizers import SGD
from lrp.LRPModel import LRPModel

img_size = 512

test_percentage = 0.05

img_size_flat = img_size * img_size * 3
img_shape_full = (img_size, img_size, 3)

replace = True

def classify_model(class_name, model_path):

    model_path = os.path.join("models", class_name, model_path)
    files = os.listdir(model_path)
    model_file_name = None
    weight_file_name = None
    for file in files:
        if len(file) > 5:
            if file[-3:] == ".h5":
                model_file_name = file
            if file[-5:] == ".hdf5":
                weight_file_name = file

    if model_file_name == None or weight_file_name == None:
        print(model_path, " model file or weight file missing")
        return

    model = load_model(os.path.join(model_path, model_file_name))
    model.load_weights(os.path.join(model_path, weight_file_name))

    model = LRPModel(model)

    if replace == False and os.path.exists(os.path.join(model_path, 'validation_results.csv')):
        return

    with open('base/Annotations/label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        all_class_samples = []
        for row in reader:
            if row[1] != class_name:
                continue
            all_class_samples.append(row)

        Y = []
        X = []
        for row in all_class_samples:
            X.append(row[0])
            Y.append(row[2].index("y"))

    num_classes = 5
    with open('base/Annotations/label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] == class_name:
                num_classes = len(row[2])
                break

    Y = to_categorical(Y, num_classes=num_classes)
    X = np.array(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_percentage, random_state=42)

    tests = []
    for index in range(X_test.shape[0]):
        image = Image.open("base/" + X_test[index])
        plt.imshow(image)
        plt.show()
        img_array = np.asarray(image)
        if img_array.shape != img_shape_full:
            image = image.resize((img_size, img_size), Image.ANTIALIAS)
            img_array = np.asarray(image)
        tests.append(img_array / 255)

        lrp_result = model.perform_lrp(np.array([img_array / 255]))
        print(lrp_result.shape)
        np.save(X_test[index].replace("/", "-"), lrp_result[0,:,:,:])
        plt.imshow(lrp_result[0,:,:,:], cmap='jet')
        plt.show()

    tests = np.array(tests)

    results = model.predict(np.array(tests), batch_size=16, verbose=0, steps=None)

    y_classes = results.argmax(axis=-1)
    print(classification_report(y_test, to_categorical(y_classes, num_classes=num_classes)))

    with open(os.path.join(model_path, 'validation_results.csv'), 'w') as wfile:
        writer = csv.writer(wfile, delimiter=',')

        for index in range(len(X_test)):
            result = results[index]
            row = [X_test[index]]
            for r in result:
                row.append(r)
            writer.writerow(row)

    print("Finished Classifying: ", model_path)

classify_model("neckline_design_labels", "InceptionV3-85")

# classes = os.listdir("models")
# for classe in classes:
#     if classe == ".DS_Store":
#         continue
#     models = os.listdir(os.path.join("models", classe))
#     for model in models:
#         if model == ".DS_Store":
#             continue
