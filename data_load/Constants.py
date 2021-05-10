import numpy as np
import pandas as pd
import pathlib
import cv2 as cv
import matplotlib.pyplot as plt

# First, we define the path and load the DF
PROJECT_PATH = str(pathlib.Path().absolute())
DF = pd.read_csv(PROJECT_PATH + "\\" + "train.csv")

# We get images shape
img = cv.imread(PROJECT_PATH + '\\train_images\\' + '04df149.jpg', -1)
HEIGHT = img.shape[0]
WIDTH = img.shape[1]
DIMENSIONS = (HEIGHT, WIDTH)

# We do some processing
corrected_df = DF['Image_Label'].str.partition('_').drop([1], axis=1).join(DF['EncodedPixels'])
corrected_df.columns = ['Image', 'Label', 'EncodedPixels']
labels = ["Fish", "Flower", "Gravel", "Sugar"]

# Get a list with all images name
image_list = np.array(list(map(lambda x: str(x), corrected_df['Image'][0:-1:4])))

# Get NotNa() elements, converts to numpy boolean array, converts the list to integer and divides by image
image_labels_one_hot = np.array(np.split(np.array(corrected_df.notna()['EncodedPixels'].to_list()).astype(int),
                                len(corrected_df) / 4))


def converter(array, labels=labels):
    boolean_array = np.array(array, dtype=bool)
    return np.array(labels)[boolean_array].tolist()


def array2string(array):
    return ", ".join(array)


# We create a dictionary with images and one-hot vector of classes
one_hot_image_dict = dict(zip(image_list, image_labels_one_hot))
values = np.array([sum(image_labels_one_hot[:, i]) for i in range(len(labels))])

hist_labels = {}
for i in range(len(image_labels_one_hot)):
    one_hot2labels = converter(image_labels_one_hot[i])
    string_labels = array2string(one_hot2labels)
    if string_labels not in hist_labels:
        hist_labels[string_labels] = 1
    else:
        hist_labels[string_labels] += 1


number_labels_image = np.array([sum(image_labels_one_hot[i]) for i in range(len(image_labels_one_hot))])
hist_number_labels = {}

for i in range(len(number_labels_image)):
    if str(number_labels_image[i]) not in hist_number_labels:
        hist_number_labels[str(number_labels_image[i])] = 1
    else:
        hist_number_labels[str(number_labels_image[i])] += 1

hist_number_labels = dict(sorted(hist_number_labels.items()))

# We get all rle based on image name
def extract_rle(filename):
    return corrected_df[corrected_df['Image'] == filename]['EncodedPixels'].fillna(-1).values


list_rle = np.split(np.array(extract_rle(corrected_df['Image'])), len(corrected_df) / 4)

# We get a dict of images and rle
rle_image_dict = dict(zip(image_list, list_rle))
