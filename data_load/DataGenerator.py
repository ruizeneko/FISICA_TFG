import copy
import numpy as np
import keras
import os
import cv2 as cv
import identification as idt


class DataGenerator(keras.utils.Sequence):
    """Data generator"""
    def __init__(self, image_list, labels, batch_size=32, dim=(32, 32, 32), n_channels=3, n_classes=4, shuffle=True,
                 base_path='C:\\FISICA_TFG\\train_images', data_frame=idt.corrected_df):
        if self.image_list is None:
            self.image_list = os.listdir(os.path.split(os.getcwd())[0] + '\\train_images')
        else:
            self.image_list = copy.deepcopy(image_list)
        # Why deep copy instead of assigment operator ?
        # We will shuffle the list -->  https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/

        self.image_list = image_list
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.data_frame = data_frame
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.base_path = base_path
        self.indexes = np.arange(len(self.image_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        images_temp = [self.image_list[k] for k in indexes]
        X, Y = self.__data_generation(images_temp)

        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, images_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, ID in enumerate(images_temp):
            img = cv.imread(self.base_path + images_temp[i])
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.
            X[i, ] = img  # X[i, ] = X[i, :, :, :]

            Y[i, ] = self.data_frame[ self.data_frame['Image'] == images_temp[i]]['EncodedPixels'].values

        return X, Y
# TODO : change data generator to create one generator for Y and another for X
# TODO : in Y generator, add directly mask
# TODO : add some data augmentation
