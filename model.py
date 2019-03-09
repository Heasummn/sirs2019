from keras.applications import (Xception, InceptionV3, VGG16, VGG19,
    ResNet50, MobileNetV2, DenseNet121, DenseNet169, DenseNet201,
    InceptionResNetV2, NASNetMobile)
from keras import Model
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D, Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import h5py
from data_gen import augment_data_gen, load_data, load_test_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as k
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import os
import config

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config = conf))

def build_model(name='vgg16', filepath=None, training=False, continuing=True):
    model = None
    base = None
    shape = config.IMAGE_DIMENSIONS

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    if os.path.exists(filepath) and training and continuing:
        model = load_model(filepath)
        return model, checkpoint, shape
    
    name = name.lower()
    if name == 'vgg16':
        base = VGG16()
    elif name == 'vgg19':
        base = VGG19()
    elif name == 'xception':
        base = Xception()
        shape = config.IMAGE_DIMENSIONS_299
    elif name == 'inceptionv3':
        base = InceptionV3()
        shape = config.IMAGE_DIMENSIONS_299
    elif name == 'resnet50':
        base = ResNet50()
    elif name == 'mobilenetv2':
        base = MobileNetV2()
    elif name == 'densenet121':
        base = DenseNet121()
    elif name == 'densenet169':
        base = DenseNet169()
    elif name == 'densenet201':
        base = DenseNet201()
    elif name == 'inceptionresnetv2':
        base = InceptionResNetV2()
        shape = config.IMAGE_DIMENSIONS_299
    elif name == 'nasnetmobile':
        base = NASNetMobile()
    elif name == 'control':
        input = Input(shape=config.IMAGE_SHAPE)
        base = Conv2D(input_shape=config.IMAGE_SHAPE, filters=16, kernel_size=3, activation='relu')(input)
        base = MaxPooling2D()(base)
        base = Flatten()(base)
        base = Model(inputs=input, output = base)

    if name != 'control':
        for layer in base.layers:
            layer.trainable = False
    
    x = Dense(1024, activation='relu')(base.output)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base.input, outputs=x)

    if os.path.exists(filepath):
        model.load_weights(filepath) 
    
    return model, checkpoint, shape

def filepath_for(name='vgg16', training=False):
    filepath = 'weights/'
    filepath += name + 'Model'

    if training:
        filepath += 'Training'
    filepath += '.hdf5'
    return filepath 

# must manually move training model to "real model"
def train_model(name='vgg16', epochs=20, continuing=True):
    filepath = filepath_for(name, True)

    model, checkpoint, shape = build_model(name, filepath, True, continuing)
    if not model._is_compiled:
        opt = Adam(lr=1e-4, decay=1e-6)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)

    data, validation = load_data()
    steps = data.shape[0]/config.BATCH_SIZE
    model.fit_generator(augment_data_gen(data, shape), validation_data=validation, epochs=epochs, steps_per_epoch=steps, callbacks=[checkpoint], class_weight={0:1.0, 1:0.4})
    del model
    del data
    del validation

def evaluate_model(name='vgg16', filename=None):
    if filename is None:
        filename = 'data/' + name + 'cm.png'
    filepath = filepath_for(name)
    model, _, _ = build_model(name, filepath)

    test_data, test_labels = load_test_data()
    preds = model.predict(test_data, batch_size=config.BATCH_SIZE)
    preds = np.argmax(preds, axis=-1)

    orig_test_labels = np.argmax(test_labels, axis=-1)
    cm  = confusion_matrix(orig_test_labels, preds)
    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.BuPu)
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.savefig(filename)

    tn, fp, fn, tp = cm.ravel()

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    print("Recall of the model is {:.2f}".format(recall))
    print("Precision of the model is {:.2f}".format(precision))
    print("f1 score is {:.2f}".format(2 * (recall * precision)/(recall + precision)))
    del model
    del test_data
    del test_labels
    del preds
    del cm  
