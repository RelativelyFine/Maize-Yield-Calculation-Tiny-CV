import numpy as np
import pandas as pd
import cv2

import zipfile
import os
import os.path
import shutil
import csv
import json

import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers

from tensorflow_addons.metrics import RSquare

import matplotlib.pyplot as plt
import cv2
import os
import scipy.io
import shutil

from sklearn.metrics import *

ratio = 1
patch_size = 100

w, h, d = 6000, 4000, 3

# list of np.array, returns list of ints


def RunOnImages(images: list) -> list:
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            padding='same', input_shape=(100, 100, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    model.summary()

    history = model.compile(optimizer='adam', loss='mean_squared_error', metrics=[
                            'mean_squared_error', RSquare()])

    checkpoint_path = "./Model/training/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    model.load_weights(checkpoint_path)

    results_arr = []

    for img_arr in images:
        img = keras.utils.array_to_img(img_arr)
        img = img.resize((int(w * ratio), int(h * ratio)))
        img_arr = keras.utils.img_to_array(img)

        ptchs = []

        for i in range(int(h * ratio / patch_size)):
            for j in range(int(w * ratio / patch_size)):

                if (100, 100, 3) != img_arr[
                    i*patch_size:i*patch_size+patch_size,
                    j*patch_size:j*patch_size+patch_size,
                    :
                ].shape:
                    continue

                ptchs.append(
                    img_arr[
                        i * patch_size: i * patch_size + patch_size,
                        j * patch_size: j * patch_size + patch_size,
                        :
                    ]
                )

        ptchs = np.array(ptchs)

        result = model.predict(ptchs)

        pd = np.sum(result*(result > 0), axis=0).astype('int')

        results_arr.append(pd)
    
    return results_arr


if __name__ == "__main__":
    img_arr = []
    for i in [
            "./Model/data/maize_tassels_counting_uav_dataset/images/DJI_0952 (2).JPG",
            "./Model/data/maize_tassels_counting_uav_dataset/images/DJI_0393 (2).JPG",
            "./Model/data/maize_tassels_counting_uav_dataset/images/DJI_0397.JPG"]:
        img_arr.append(keras.utils.load_img(i))

    print(RunOnImages(img_arr))

    # size_arr = []
    # for i in [
    #         "./data/maize_tassels_counting_uav_dataset/labels/DJI_0952 (2).csv",
    #         "./data/maize_tassels_counting_uav_dataset/labels/DJI_0393 (2).csv",
    #         "./data/maize_tassels_counting_uav_dataset/labels/DJI_0397.csv"]:
    #     size_arr.append(sum(1 for line in open(i)) - 2)

    # print(size_arr)
    
