import numpy as np
import cv2 as cv
import pandas as pd
import segmentation as sgt

import matplotlib.pyplot as plt

import pathlib

PROJECT_PATH = str(pathlib.Path().absolute())
PATTERN = sgt.PATTERN
COLUMN_TAGS = sgt.COLUMN_TAGS
DF = pd.read_csv(PROJECT_PATH + "\\" + "train.csv")

corrected_df = DF['Image_Label'].str.partition('_').drop([1], axis=1).join(DF['EncodedPixels'])
corrected_df.columns = ['Image', 'Label', 'EncodedPixels']

colors = {'Fish': 'red', 'Flower': 'blue', 'Gravel': 'purple', 'Sugar': 'green'}
code = {'red': (255, 0, 0), 'blue': (0, 0, 255), 'purple': (255, 0, 255), 'green': (127, 255, 0)}


def get_min(array):
    min_y = 10e4
    pos_y = 0

    if len(array) <= 1:
        array = array[0]
        for i in range(len(array)):
            if array[i][0][1] < min_y:
                min_y = array[i][0][1]
                pos_y = i
        return array[pos_y][0][0], min_y

    else:
        index_aux = 0
        for i in range(len(array)):
            for j in range(len(array[i])):
                if array[i][j][0][1] < min_y:
                    min_y = array[i][j][0][1]
                    pos_y = j
                    index_aux = i
        return array[index_aux][pos_y][0][0], min_y


def get_true_index(img_name):
    return corrected_df[corrected_df['Image'].str.contains(img_name)].index.tolist()


def get_rle_label(img_name, shapes):
    cleaned_df = corrected_df.iloc[get_true_index(img_name)].drop(['Image'], axis=1).fillna(-1)
    img_map = dict()
    for ind in cleaned_df.index:
        if cleaned_df['EncodedPixels'][ind] == -1:
            img_map[cleaned_df['Label'][ind]] = -1
        else:
            img_map[cleaned_df['Label'][ind]] = sgt.rle_to_mask(cleaned_df['EncodedPixels'][ind], shapes[1],
                                                                shapes[0])
    return img_map


def contour_label_img(img_name, img, tag):
    color = code[colors[tag]]
    cnt = get_rle_label(img_name, img.shape[0:2])[tag]

    contours, _ = cv.findContours(cnt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 3)

    left_top_corner = get_min(contours)
    cv.putText(img, tag, (left_top_corner[0] + 25, left_top_corner[1] + 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)


def counter_img(img_name):
    img = cv.imread(PROJECT_PATH + '\\train_images\\' + img_name, -1)
    cnt = get_rle_label(img_name, img.shape[0:2])

    for ind in cnt:
        if type(cnt[ind]) is not int:
            contour_label_img(img_name, img, ind)

    im_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return im_rgb


def various_counter_img(height=3, width=3):
    np.random.seed(50)

    fig, axs = plt.subplots(height, width, figsize=(20, 20))
    number_images_shown = height * width

    # create a list of random indices
    rnd_indices_id = [np.random.choice(range(0, len(corrected_df['Image']))) for _ in range(height * width)]

    for im in range(0, number_images_shown):
        # open image with a random index

        i = im // width
        j = im % width

        # plot the image
        img_name = corrected_df['Image'][rnd_indices_id[im]]
        axs[i, j].imshow(counter_img(img_name))  # plot the data
        axs[i, j].axis('off')
        axs[i, j].set_title(sgt.get_labels(img_name))

    # set subtitle
    plt.suptitle('Maps and represented clouds')
    plt.show()
