# import basic libraries
from glob import glob

import pathlib

import pandas as pd
import numpy as np

# import plotting
from matplotlib import pyplot as plt

# import image manipulation
from PIL import Image

from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.imgaug

PROJECT_PATH = str(pathlib.Path().absolute()) + "\\"
PATTERN = '[_]+'
COLUMN_TAGS = ["Image_Label", "EncodedPixels"]
DF = pd.read_csv(PROJECT_PATH + "\\" + "train.csv")

corrected_df = DF['Image_Label'].str.partition('_').drop([1], axis=1).join(DF['EncodedPixels'])
corrected_df.columns = ['Image', 'Label', 'EncodedPixels']


def get_labels(img_name):
    corrected_df_aux = corrected_df.dropna().reset_index(drop=True)
    image_labels_index = corrected_df_aux['Image'].index[corrected_df_aux['Image'] == img_name].tolist()
    return ', '.join(corrected_df_aux['Label'][image_labels_index])


def plot_training_images(project_path, width=3, height=3):
    """
    Function to plot grid with several examples of cloud images from train set.
    INPUT:
        width - number of images per row
        height - number of rows

    OUTPUT: None
    """
    # get a list of images from training set
    images = sorted(glob(project_path + "train_images/" + '*.jpg'))
    fig, axs = plt.subplots(height, width, figsize=(width * 3, height * 3))
    number_images_shown = height * width

    # create a list of random indices
    rnd_indices = np.random.randint(0, len(images), size=number_images_shown)

    for im in range(0, number_images_shown):
        # open image with a random index
        image = Image.open(images[rnd_indices[im]])
        img_name = corrected_df['Image'][rnd_indices[im]]

        i = im // width
        j = im % width

        # plot the image
        axs[i, j].imshow(image)  # plot the data
        axs[i, j].axis('off')
        axs[i, j].set_title(get_labels(img_name))

    # set subtitle
    plt.suptitle('Sample images from the train set')
    plt.show()


# def hist_combinations(labels_dict):
#     """
#     Function to plot an histogram with the combinations of labels that appear
#     INPUT:
#         labels_dict: dictionary with the image and the corresponding labels
#     OUTPUT: histogram with the combinations of labels that appear
#     """
#     combinations = dict()
#     for key, value in labels_dict.items():
#         if list2string(value) not in combinations:
#             combinations[list2string(value)] = 1
#         else:
#             combinations[list2string(value)] += 1
#
#     plt.xticks(rotation='vertical')
#     plt.bar(combinations.keys(), combinations.values())
#     plt.show()
#
#

def rle_to_mask(rle_string, width, height):
    """
    convert RLE(run length encoding) string to numpy array

    Parameters:
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask

    Returns:
    numpy.array: numpy array of the mask
    """
    rows, cols = height, width

    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index + length] = 255
        img = img.reshape(cols, rows)
        img = img.T
        return img


def map_label(img_name):
    index = corrected_df['Image'].index[corrected_df['Image'] == img_name].tolist()
    return corrected_df.iloc[index]


def get_map_label(img_name, label):
    label_map = map_label(img_name)['Label']
    index = label_map.index[label_map == label].tolist()
    return corrected_df['EncodedPixels'][index].fillna('-1').tolist()


def get_mask_filename(img_name, label, shape=(2100, 1400)):
    # convert rle to mask
    rle = get_map_label(img_name, label)[0]

    if rle != '-1':
        mask_label = rle_to_mask(rle, shape[0], shape[1])
        mask_label = np.clip(mask_label, 0, 1)
    else:
        # empty mask
        mask_label = np.zeros((shape[0], shape[1]), dtype=np.uint8).T

    return mask_label


def draw_labels(image, np_mask, label):
    """
    Function to add labels to the image.
    """
    if np.sum(np_mask) > 0:
        x, y = np.argwhere(np_mask == 1)[0]
        image = imgaug.imgaug.draw_text(image, x, y, label, color=(255, 255, 255), size=50)

    return image


def segmentation_map_image(img_name):
    # open the image
    image = np.asarray(Image.open(PROJECT_PATH + "train_images\\" + img_name))

    fish_mask = get_mask_filename(img_name, 'Fish')
    flower_mask = get_mask_filename(img_name, 'Flower')
    gravel_mask = get_mask_filename(img_name, 'Gravel')
    sugar_mask = get_mask_filename(img_name, 'Sugar')

    segmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)

    segmap = np.where(fish_mask == 1, 1, segmap)
    segmap = np.where(flower_mask == 1, 2, segmap)
    segmap = np.where(gravel_mask == 1, 3, segmap)
    segmap = np.where(sugar_mask == 1, 4, segmap)

    segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

    image = np.asarray(segmap.draw_on_image(np.asarray(image))).reshape(np.asarray(image).shape)

    image = draw_labels(image, fish_mask, 'Fish')
    image = draw_labels(image, flower_mask, 'Flower')
    image = draw_labels(image, gravel_mask, 'Gravel')
    image = draw_labels(image, sugar_mask, 'Sugar')

    return image


# helper function to visualize several segmentation maps on a single image
def visualize_several_maps(img_name):
    """
    Function to visualize several segmentation maps.
    INPUT:
        image_id - filename of the image
    """
    # draw segmentation maps and labels on image
    image = segmentation_map_image(img_name)

    labels = get_labels(img_name)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis('off')
    plt.title('Segmentation maps: ' + labels)
    ax.imshow(image)


def multiple_seg_map(width=2, height=3):
    # get a list of images from training set
    images = sorted(glob(PROJECT_PATH + "train_images/" + '*.jpg'))
    fig, axs = plt.subplots(height, width, figsize=(20, 20))
    number_images_shown = height * width

    # create a list of random indices
    rnd_indices_seg = [np.random.choice(range(0, len(images))) for _ in range(height * width)]
    for im in range(0, number_images_shown):
        # open image with a random index

        i = im // width
        j = im % width

        # draw segmentation maps and labels on image
        img_name_eventual = images[rnd_indices_seg[im]].split('\\')[-1]
        image = segmentation_map_image(img_name_eventual)

        labels = get_labels(img_name_eventual)

        # plot the image
        axs[i, j].imshow(image)  # plot the data
        axs[i, j].axis('off')
        axs[i, j].set_title(labels)

    # set subtitle
    plt.suptitle('Maps and represented clouds')
    plt.show()


# TODO : CREATE BOUNDING BOX
# TODO : IMPROVE TWO CLOUDS OVERLAPPING
