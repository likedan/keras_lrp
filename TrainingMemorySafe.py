import numpy as np
import csv, os, sys
from PIL import Image
import datetime

from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

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
        self.epoch = 0
        self.num_classes = num_classes
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch % self.N == 0 and self.N != 0:
            y_prob = self.model.predict(self.X_test)
            y_classes = y_prob.argmax(axis=-1)
            print(classification_report(self.y_test, to_categorical(y_classes, num_classes=self.num_classes)))
        self.epoch += 1

class Trainer:

    def __init__(self, model_name="Xception", train_class_name=None, training_batch_size=100, test_percentage=0.02, learning_rate=0.0001, validation_every_X_batch=5, saving_frequency=1, gpu_num=1, dropout=0.2):

        if train_class_name == None:
            print("You must specify train_class_name")
            return

        self.save_frequency = saving_frequency
        self.validation_every_X_batch = validation_every_X_batch
        self.model_file = model_name + "-{date:%Y-%m-%d-%H-%M-%S}".format( date=datetime.datetime.now())
        print("model_folder: ", self.model_file)

        self.train_class_name = train_class_name
        if not os.path.exists(os.path.join("models", train_class_name)):
            os.makedirs(os.path.join("models", train_class_name))

        self.training_batch_size = training_batch_size

        # We know that MNIST images are 28 pixels in each dimension.
        img_size = 512
        self.img_size = img_size
        self.img_size_flat = img_size * img_size * 3

        self.img_shape_full = (img_size, img_size, 3)

        self.test = {}

        with open('base/Annotations/label.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[1] == self.train_class_name:
                    self.num_classes = len(row[2])
                    break

        # Start construction of the Keras Sequential model.
        input_tensor = Input(shape=self.img_shape_full)

        if model_name == "Xception":
            base_model = xception.Xception(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)
        elif model_name == "VGG16":
            base_model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)
        elif model_name == "VGG19":
            base_model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)
        elif model_name == "DenseNet121":
            base_model = densenet.DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)
        elif model_name == "DenseNet201":
            base_model = densenet.DenseNet201(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)
        elif model_name == "ResNet50":
            base_model = resnet50.ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)
        elif model_name == "InceptionV3":
            base_model = inception_v3.InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)            
        elif model_name == "InceptionResNetV2":
            base_model = inception_resnet_v2.InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=self.num_classes)

        x = base_model.output
        x = Dropout(dropout)(x)
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        if gpu_num > 1:
            model = multi_gpu_model(model, gpus=gpu_num)
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

            self.Y = []
            self.X = []
            test_count = int(test_percentage * len(all_class_samples))
            print("Training " + train_class_name + " with: " + str(int((1 - test_percentage) * len(all_class_samples))) + ", Testing with: " + str(test_count), str(self.num_classes), "Classes")
            print("Loading images...")
            for row in all_class_samples:
                self.X.append(row[0])
                self.Y.append(row[2].index("y"))

        self.Y = to_categorical(self.Y, num_classes=self.num_classes)
        self.X = np.array(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size = test_percentage, random_state=42)
        class_weight = {}
        class_count = np.sum(self.y_train, axis=0)
        print("Training Sample for each Class", class_count)
        for class_index in range(self.num_classes):
            class_weight[class_index] = 1 / class_count[class_index] * len(all_class_samples) / self.num_classes
        self.class_weight = class_weight
        print("Class weights: ", self.class_weight)
        os.makedirs(os.path.join("models", train_class_name, self.model_file))
        model.save(os.path.join("models", train_class_name, self.model_file, train_class_name + "_" + "model.h5"))

        self.X_T = []
        for index in range(self.X_test.shape[0]):
            image = Image.open("base/" + self.X_test[index])
            img_array = np.asarray(image)
            if img_array.shape != self.img_shape_full:
                image = image.resize((img_size, img_size), Image.ANTIALIAS)
                img_array = np.asarray(image)
            self.X_T.append(img_array / 255)
        self.X_T = np.array(self.X_T)

    def train(self, epochs=100):
        checkpoint = ModelCheckpoint(os.path.join("models", self.train_class_name, self.model_file, "weights.hdf5"), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        self.model.fit_generator(self.generate_arrays_from(), class_weight=self.class_weight, steps_per_epoch=int(self.X_train.shape[0] / self.training_batch_size * self.save_frequency), epochs=epochs, validation_data=(self.X_T, self.y_test), callbacks=[Validation(self.model, self.validation_every_X_batch, self.num_classes, self.X_T, self.y_test), checkpoint])

    def generate_arrays_from(self):
        Y = []
        X = []
        while 1:
            for index in range(self.X_train.shape[0]):
                image = Image.open("base/" + self.X_train[index])
                img_array = np.asarray(image)
                if img_array.shape != self.img_shape_full:
                    image = image.resize((self.img_size, self.img_size), Image.ANTIALIAS)
                    img_array = np.asarray(image)

                X.append(img_array)
                Y.append(self.y_train[index])

                if index % self.training_batch_size == 0:
                    x, y = np.array(X) / 255, np.array(Y)
                    yield (x, y)
                    Y = []
                    X = []
            if len(X) > 0:
                x, y = np.array(X) / 255, np.array(Y)
                yield (x, y)
                Y = []
                X = []
