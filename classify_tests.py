# go over the models dir and classify everything

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv, os
import math
from PIL import Image

from keras.models import load_model
from tensorflow.python.keras.models import load_model
from lrp.LRPModel import LRPModel

img_size = 512

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

    if replace == False and os.path.exists(os.path.join(model_path, 'test_results_formatted.csv')):
        return

    tests = []
    rows = []
    with open('rank/Tests/question.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] != class_name:
                continue
            image = Image.open("rank/" + row[0])
            plt.imshow(image)
            plt.show()
            img_array = np.asarray(image)
            if img_array.shape != img_shape_full:
                image = image.resize((img_size, img_size), Image.ANTIALIAS)
                img_array = np.asarray(image)
            tests.append(img_array/ 255)

            lrp_result = model.perform_lrp(np.array([img_array / 255]))
            print(lrp_result.shape)
            np.save(row[0].replace("/", "-"), lrp_result[0, :, :, :])
            plt.imshow(lrp_result[0, :, :, :], cmap='jet')
            plt.show()

            rows.append(row)
    results = model.predict(np.array(tests), batch_size=16, verbose=0, steps=None)

    with open(os.path.join(model_path, 'test_results_formatted.csv'), 'w') as wfile:
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
            writer.writerow(row)

    with open(os.path.join(model_path, 'test_results_raw.csv'), 'w') as wfile:
        writer = csv.writer(wfile, delimiter=',')

        for index in range(len(rows)):
            result = results[index]
            row = rows[index]
            del row[-1]
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
#         classify_model(classe, model)