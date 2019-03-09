from os import listdir
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import cv2
import imgaug.augmenters as augs
from config import *

train_data_directory = './chest_xray/train/'
test_data_dir = './chest_xray/test/'
validation_data_dir = './chest_xray/val/'

def load_data():
    training = []

    # load array with training data and classification: 0=NORMAL, 1=PNEUMONIA
    for img in filter((lambda x: x.endswith(".jpeg")), listdir(train_data_directory + 'NORMAL')):
        training.append( (train_data_directory + 'NORMAL/' + img, 0) )

    for img in filter((lambda x: x.endswith(".jpeg")), listdir(train_data_directory + 'PNEUMONIA')):
        training.append( (train_data_directory + 'PNEUMONIA/' + img, 1) )

    training = pd.DataFrame(training, columns=['image', 'type'], index=None)
    training = training.sample(frac=1).reset_index(drop=True)

    val = []
    val_labels = []
    for img in filter((lambda x: x.endswith(".jpeg")), listdir(validation_data_dir + 'NORMAL')):
        img = cv2.imread(validation_data_dir + 'NORMAL/' + img)
        img = cv2.resize(img, IMAGE_DIMENSIONS)
        if img.shape[2]==1:
                img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255
        label = to_categorical(0, num_classes=2)
        val.append(img)
        val_labels.append(label)

    for img in filter((lambda x: x.endswith(".jpeg")), listdir(validation_data_dir + 'PNEUMONIA')):
        img = cv2.imread(validation_data_dir + 'PNEUMONIA/' + img)
        img = cv2.resize(img, IMAGE_DIMENSIONS)
        if img.shape[2]==1:
                img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255
        label = to_categorical(1, num_classes=2)
        val.append(img)
        val_labels.append(label)

    validation = (np.array(val), np.array(val_labels))

    return (training, validation)

def load_test_data():
    test = []
    test_labels = []
    for img in filter((lambda x: x.endswith(".jpeg")), listdir(test_data_dir + 'NORMAL')):
        img = cv2.imread(test_data_dir + 'NORMAL/' + img)
        img = cv2.resize(img, IMAGE_DIMENSIONS)
        if img.shape[2]==1:
                img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255
        label = to_categorical(0, num_classes=2)
        test.append(img)
        test_labels.append(label)

    for img in filter((lambda x: x.endswith(".jpeg")), listdir(test_data_dir + 'PNEUMONIA')):
        img = cv2.imread(test_data_dir + 'PNEUMONIA/' + img)
        img = cv2.resize(img, IMAGE_DIMENSIONS)
        if img.shape[2]==1:
                img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255
        label = to_categorical(1, num_classes=2)
        test.append(img)
        test_labels.append(label)
    
    testing = (np.array(test), np.array(test_labels))

    return testing

def augment_data_gen(data, img_size):
    length = len(data)
    total_batches = length/BATCH_SIZE
    new_data = np.zeros((BATCH_SIZE,) + img_size + (3,), dtype=np.float32)
    labels = np.zeros( (BATCH_SIZE, 2), dtype=np.float32 )

    indices = np.arange(length)

    batch = 0
    

    augmentor = augs.OneOf([
        augs.Fliplr(1.0),
        augs.Multiply((1.2, 1.5)),
        augs.Affine(
            rotate=(-20, 20)
        ),
        augs.AdditiveGaussianNoise(scale=(0.0, 0.05*255))
    ])    

    # create a generator
    while True:
        np.random.shuffle(indices)
        
        current_batch = indices[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
        count = 0

        for index in current_batch:
            
            name = data.iloc[index]['image']
            label = data.iloc[index]['type']

            encoded = to_categorical(label, num_classes=2)
            
            img = cv2.imread(name)
            img = cv2.resize(img, img_size)

            if img.shape[2]==1:
                img = np.dstack([img, img, img])
            
            # make RGB and normalize
            original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original = img.astype(np.float32)/255

            new_data[count] = original
            labels[count] = encoded

            if label==0 and count < BATCH_SIZE - 3:
                aug1 = augmentor.augment_image(img)
                aug2 = augmentor.augment_image(img)

                aug1 = cv2.cvtColor(aug1, cv2.COLOR_BGR2RGB)
                aug1 = aug1.astype(np.float32)/255
                aug2 = cv2.cvtColor(aug2, cv2.COLOR_BGR2RGB)
                aug2 = aug2.astype(np.float32)/255
                
                new_data[count + 1] = aug1
                labels[count + 1] = encoded
                new_data[count + 2] = aug2
                labels[count + 2] = encoded
                count += 2
            
            if count == BATCH_SIZE - 1:
                break

            count += 1
        batch += 1
        yield new_data, labels
        if batch >= total_batches:
            batch = 0

