import numpy as np
import csv, os, sys
from PIL import Image
import datetime

from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras import optimizers
from sklearn.metrics import classification_report
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121

from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *

class Validation(Callback):
    def __init__(self, model, N, num_classes, X_test, y_test):
        self.model = model
        self.N = N
        self.batch = 0
        self.num_classes = num_classes
        self.X_test = X_test
        self.y_test = y_test

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0 and self.N != 0:
            y_prob = self.model.predict(self.X_test)
            y_classes = y_prob.argmax(axis=-1)
            print(classification_report(self.y_test, to_categorical(y_classes, num_classes=self.num_classes)))
        self.batch += 1

class Trainer:

    def __init__(self, model_name="Xception", train_class_name=None, training_batch_size=100, existing_weight=None, test_percentage=0.02, learning_rate=0.000, validation_every_X_batch=5):

        if train_class_name == None:
            print("You must specify train_class_name")
            return

        self.validation_every_X_batch = validation_every_X_batch
        self.Y = []

        self.model_file = model_name + "-{date:%Y-%m-%d-%H-%M-%S}".format( date=datetime.datetime.now())
        print("model_folder: ", self.model_file)

        self.train_class_name = train_class_name
        if not os.path.exists(os.path.join("models", train_class_name)):
            os.makedirs(os.path.join("models", train_class_name))

        self.training_batch_size = training_batch_size

        # We know that MNIST images are 28 pixels in each dimension.
        img_size = 512

        self.img_size_flat = img_size * img_size * 3

        self.img_shape_full = (img_size, img_size, 3)

        self.test = {}

        with open('base/Annotations/label.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            all_class_samples = []
            for row in reader:
                if row[1] == self.train_class_name:
                    self.num_classes = len(row[2])
                    break

        # Start construction of the Keras Sequential model.
        input_tensor = Input(shape=self.img_shape_full)
        if model_name == "Xception":
            base_model = Xception(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)
        elif model_name == "MobileNet":
            base_model = MobileNet(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)
        elif model_name == "DenseNet121":
            base_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)


        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        model = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        self.model = model
        print(model.summary())

        self.optimizer = optimizers.Adam(lr=learning_rate)

        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


        with open('base/Annotations/label.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            all_class_samples = []
            for row in reader:
                if row[1] != self.train_class_name:
                    continue
                all_class_samples.append(row)

            self.X = np.zeros((len(all_class_samples), img_size, img_size, 3))
            # self.X = np.zeros((10, img_size, img_size, 3))
            test_count = int(test_percentage * len(all_class_samples))
            index = 0
            print("Training " + train_class_name + " with: " + str(int((1 - test_percentage) * len(all_class_samples))) + ", Testing with: " + str(test_count), str(self.num_classes), "Classes")
            print("Loading images...")
            for row in all_class_samples:
                image = Image.open("base/" + row[0])
                img_array = np.asarray(image)
                if img_array.shape != self.img_shape_full:
                    image = image.resize((img_size, img_size), Image.ANTIALIAS)
                    img_array = np.asarray(image)
                self.X[index] = img_array
                self.Y.append(row[2].index("y"))
                if index % 500 == 0:
                    print(index)
                index += 1

        self.Y = to_categorical(self.Y, num_classes=self.num_classes)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size = test_percentage, random_state=42)
        del self.X
        class_weight = {}
        class_count = np.sum(self.y_train, axis=0)
        print("Training Sample for each Class", class_count)
        for class_index in range(self.num_classes):
            class_weight[class_index] = 1 /(class_count[class_index] / np.sum(class_count)) / self.num_classes
        self.class_weight = class_weight
        print("Class weights: ", self.class_weight)
        os.makedirs(os.path.join("models", train_class_name, self.model_file))
        model.save(os.path.join("models", train_class_name, self.model_file, train_class_name + "_" + "model.h5"))

    def train(self, steps_per_epoch=10, epochs=100):
        checkpoint = ModelCheckpoint(os.path.join("models", self.train_class_name, self.model_file, "weights.hdf5"), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        self.model.fit(self.X_train, self.y_train, class_weight=self.class_weight, batch_size=self.training_batch_size, epochs=epochs, validation_data=(self.X_test, self.y_test), callbacks=[checkpoint, Validation(self.model, self.validation_every_X_batch, self.num_classes, self.X_test, self.y_test)])





