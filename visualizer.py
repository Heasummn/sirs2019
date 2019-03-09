import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_gen import load_data, augment_data_gen
import cv2

def visualize_counts(training):
    print(training.head())

    count = training['type'].value_counts()
    print(count)
    plt.bar(count.index, count.values)
    plt.xticks(range(len(count.index)), ['Normal', 'Pneumonia'])

    plt.show()


def visualize_images(training):
    pnu_sample = list(training[training['type']==1]['image'][:5])
    norm_sample = list(training[training['type']==0]['image'][:5])

    image_samples = norm_sample + pnu_sample

    _, axes = plt.subplots(2, 5)
    for i in range(10): 
        img = cv2.imread(image_samples[i], 0) 
        axes[i//5, i%5].imshow(img, cmap='gray')
        if i>=5:
            axes[i//5, i%5].set_title("Pneumonia")
        else:
            axes[i//5, i%5].set_title("Normal")
        axes[i//5, i%5].axis('off')
        axes[i//5, i%5].set_aspect('auto')

    plt.show()

def show_images(images):
    row = len(images)//2
    _, axes = plt.subplots(2, row)
    
    for i, img in enumerate(images):
        axes[i//row, i%row].imshow(img, cmap='gray')
        
        axes[i//row, i%row].axis('off')
        axes[i//row, i%row].set_aspect('auto')

    plt.show()

def augment_infinite():
    gen = augment_data_gen(load_data()[0])

    for images, _ in gen:
        show_images(images)

augment_infinite()
