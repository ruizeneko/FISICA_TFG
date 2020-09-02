import numpy as np
import cv2 as cv
import tensorflow.keras
import tools


class DataGenerator(tensorflow.keras.utils.Sequence):
    """
    aaa
    """
    def __init__(self, dataframe, directory, batch_size, n_channels, target_size, n_classes,
                 mode='fit', target_df=None, shuffle=True, preprocessing=None, augmentation=None, seed=0):

        self.batch_size = batch_size
        self.dataframe = dataframe
        self.mode = mode
        self.directory = directory
        self.target_df = target_df
        self.target_size = target_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.seed = seed
        self.mask_shape = (1400, 2100)
        self.list_IDs = self.dataframe.index

        if self.seed is not None:
            np.random.seed(self.seed)

        self.on_epoch_end()

    def __len__(self):
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        X = self.__generate_x(list_IDs_batch)

        if self.mode == 'fit':
            Y = self.__generate_y(list_IDs_batch)

            if self.augmentation:
                X, Y = self.__augment_batch(X, Y)

            return X, Y

        elif self.mode == 'predict':
            return X

    def on_epoch_end(self):
        """
        aaa
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __generate_x(self, list_IDs_batch):
        X = np.empty((self.batch_size, *self.target_size, self.n_channels))

        for i, ID in enumerate(list_IDs_batch):
            img_name = self.dataframe['image'].loc[ID]
            img_path = self.directory + img_name
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            if self.preprocessing:
                img = self.preprocessing(img)

            X[i, ] = img

        return X

    def __generate_y(self, list_IDs_batch):
        Y = np.empty((self.batch_size, *self.target_size, self.n_classes), dtype=int)

        for i, ID in enumerate(list_IDs_batch):
            img_name = self.dataframe['image'].loc[ID]
            image_df = self.target_df[self.target_df['image'] == img_name]
            rles = image_df['EncodedPixels'].values
            masks = tools.get_masks(rles, input_shape=self.mask_shape, reshape=self.target_size)
            Y[i, ] = masks

        return Y

    def __augment_batch(self, X_batch, Y_batch):
        for i in range(X_batch.shape[0]):
            X_batch[i, ], Y_batch[i, ] = self.__random_transform(X_batch[i, ], Y_batch[i, ])

        return X_batch, Y_batch

    def __random_transform(self, X, Y):
        composed = self.augmentation(image=X, mask=Y)
        X_aug = composed['image']
        Y_aug = composed['mask']

        return X_aug, Y_aug
