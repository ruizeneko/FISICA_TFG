"""Some tools used later"""
import os
import cv2 as cv
import numpy as np
from keras import backend as K
import math
import matplotlib.pyplot as plt
from data_load import DataGenerator
from data_load import Constants
import seaborn as sns
import pandas as pd


def get_train_val(path):
    """
    Extract the directory under study
    Parameters:
        path : whole path of the directory
    Return:
        string: the directory we are creating (train/val/test)
    """
    sub_string = path.split('/')[3]
    return sub_string.split('_')[0].capitalize()


def check_if_exist(list_files, path):
    """
    Check if N (here N = 50) images exist in the train/test/val directory. If they exist, do not move them again. If
    they do not, they move.
    Parameters:
        list_files: the list of files to check
        path: where the files are
    Return:
        boolean: if all the files exist or not
    """
    for i in range(len(list_files)):
        if os.path.isdir('/kaggle/working/' + path):
            if not os.path.isfile('/kaggle/working/' + path + '/' + list_files[i]):
                print("Files do not match. Creating {} directory...".format(get_train_val('/kaggle/working/' + path)))
                return False
    print('Files do exist in {} directory.'.format(get_train_val('/kaggle/working/' + path)))
    return True


def get_resize_image(img_name, shape, test_train):
    """
    Resizes and changes from BGR to RGB an image opened by OpenCV
    Parameters:
        img_name (string): the name of the image
        shape (list_int): the shape of the future image
        test_train (string): the subset the image belongs to
    Returns:
        numpy-array: the img converted
    """
    if test_train == 'train':
        img = cv.imread(Constants.PROJECT_PATH + '/train_images/' + img_name)
    elif test_train == 'test':
        img = cv.imread(Constants.PROJECT_PATH + '/test_images/' + img_name)

    try:
        assert isinstance(img, np.ndarray)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        resized_img = cv.resize(img, shape)
        return resized_img
    except AssertionError:
        print("Problems loading the image. Check both path and image name")


def decode_rle(mask_rle, shape=(1400, 2100)):
    """
    Converts
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction


def resize(img, input_shape):
    """
    Reshape a numpy array, which is input_shape=(height, width),
    as opposed to input_shape=(width, height) for cv2
    """
    height, width = input_shape
    return cv.resize(img, (width, height))


def move_data(image_df, path, HEIGHT=384, WIDTH=480, test_train='train'):
    """
    Creates an image in a given directory
    Parameters:
        image_df: the dataframe of images
        path: where the images are
        HEIGHT: the dimensions of the image
        WIDTH: the dimensions of the image
        test_train: the dataset image belongs to
    """
    for i in range(image_df.shape[0]):
        item = image_df.iloc[i]
        img_name = item['image']
        image = get_resize_image(img_name, (WIDTH, HEIGHT), test_train)
        cv.imwrite(path + img_name, image)


def rle_to_mask(rle, input_shape):
    """
    convert RLE(run length encoding) string to numpy array

    Parameters:
    rle (str): string of rle encoded mask
    input_shape(numpy_int): height and width of the mask


    Returns:
    numpy-array: numpy array of the mask
    """
    width, height = input_shape[:2]
    mask = np.zeros(width * height).astype(np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return mask.reshape(height, width).T


def get_masks(rle_list, input_shape, reshape=None):
    """
    Return all the mask form an image
    Parameters:
        rle_list: the list of the mage
        input_shape: the masks shape
        reshape: reshape or not
    Returns:
        numpy-array: all the masks
    """
    n_mask = len(rle_list)
    if reshape is None:
        masks = np.zeros((*input_shape, n_mask))
    else:
        masks = np.zeros((*reshape, n_mask))
    for i, rle in enumerate(rle_list):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle_to_mask(rle, input_shape)
            else:
                mask = rle_to_mask(rle, input_shape)
                reshape_mask = resize(mask, reshape)
                masks[:, :, i] = reshape_mask
    return masks


def dice_coefficient(y_true, y_pred):
    """The metrics.
    For further information refer to: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Parameters:
        y_true: true label
        y_pred: predicted label
    Returns
        double: the result
    """
    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)
    intersection = np.logical_and(y_true, y_pred)
    return (2. * intersection.sum()) / (y_true.sum() + y_pred.sum())


def dice_coef(y_true, y_pred, smooth=1):
    """The metrics.
    For further information refer to: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Parameters:
        y_true: true label
        y_pred: predicted label
        smooth: smooth of the metric
    Returns
        double: the result
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Model evaluation
def plot_metrics(history, metric_list=None):
    """
    aaa
    """
    if metric_list is None:
        metric_list = ['loss', 'dice_coef']
    fig, axes = plt.subplots(len(metric_list), sharex='col', figsize=(22, len(metric_list) * 4))
    axes = axes.flatten()

    for index, metric in enumerate(metric_list):
        axes[index].plot(history[metric], label='Train %s' % metric)
        axes[index].plot(history['val_%s' % metric], label='Validation %s' % metric)
        axes[index].legend(loc='best')
        axes[index].set_title(metric)

    plt.xlabel('Epochs')
    sns.despine()
    plt.show()


# Model post process
def post_process(probability, threshold=0.5, min_size=10000):
    """
    aaa
    """
    mask = cv.threshold(probability, threshold, 1, cv.THRESH_BINARY)[1]
    num_component, component = cv.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(probability.shape, np.float32)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
    return predictions

    # Prediction evaluation


def get_metrics(model, target_df, df, df_images_dest_path, thresholds, min_mask_sizes, N_CLASSES=4,
                preprocessing=None, set_name='Complete set'):
    """
    aaa
    """
    class_names = ['Fish', 'Flower', 'Gravel', 'Sugar']
    metrics = []

    for class_name in class_names:
        metrics.append([class_name, 0, 0])

    metrics_df = pd.DataFrame(metrics, columns=['Class', 'Dice', 'Dice Post'])

    for i in range(0, df.shape[0], 180):
        batch_idx = list(range(i, min(df.shape[0], i + 180)))
        batch_set = df[batch_idx[0]: batch_idx[-1] + 1]
        ratio = len(batch_set) / len(df)

        generator = DataGenerator.DataGenerator(
            directory=df_images_dest_path,
            dataframe=batch_set,
            target_df=target_df,
            batch_size=len(batch_set),
            target_size=model.input_shape[1:3],
            n_channels=model.input_shape[3],
            n_classes=N_CLASSES,
            preprocessing=preprocessing,
            shuffle=False)

        x, y = generator.__getitem__(0)
        predictions = model.predict(x)

        for class_index in range(N_CLASSES):
            class_score = []
            class_score_post = []
            mask_class = y[..., class_index]
            pred_class = predictions[..., class_index]
            for index in range(len(batch_idx)):
                sample_mask = mask_class[index, ]
                sample_pred = pred_class[index, ]
                sample_pred_post = post_process(sample_pred, threshold=thresholds[class_index],
                                                min_size=min_mask_sizes[class_index])
                if (sample_mask.sum() == 0) & (sample_pred.sum() == 0):
                    dice_score = 1.
                else:
                    dice_score = dice_coefficient(sample_pred, sample_mask)
                if (sample_mask.sum() == 0) & (sample_pred_post.sum() == 0):
                    dice_score_post = 1.
                else:
                    dice_score_post = dice_coefficient(sample_pred_post, sample_mask)
                class_score.append(dice_score)
                class_score_post.append(dice_score_post)
            metrics_df.loc[metrics_df['Class'] == class_names[class_index], 'Dice'] += np.mean(class_score) * ratio
            metrics_df.loc[metrics_df['Class'] == class_names[class_index], 'Dice Post'] += np.mean(
                class_score_post) * ratio

    metrics_df = metrics_df.append({'Class': set_name, 'Dice': np.mean(metrics_df['Dice'].values),
                                    'Dice Post': np.mean(metrics_df['Dice Post'].values)}, ignore_index=True).set_index(
        'Class')

    return metrics_df


def inspect_predictions(df, image_ids, images_dest_path, pred_col=None, label_col='EncodedPixels',
                        title_col='Image_Label', img_shape=(525, 350), figsize=(22, 6)):
    """
    aaa
    """
    if pred_col:
        for sample in image_ids:
            sample_df = df[df['image'] == sample]
            fig, axes = plt.subplots(2, 5, figsize=figsize)
            img = cv.imread(images_dest_path + sample_df['image'].values[0])
            img = cv.resize(img, img_shape)
            axes[0][0].imshow(img)
            axes[1][0].imshow(img)
            axes[0][0].set_title('Label', fontsize=16)
            axes[1][0].set_title('Predicted', fontsize=16)
            axes[0][0].axis('off')
            axes[1][0].axis('off')
            for i in range(4):
                mask = sample_df[label_col].values[i]
                try:
                    mask = decode_rle(mask)
                except math.isnan(mask):
                    mask = decode_rle(mask)
                    math.isnan(mask)
                axes[1][i + 1].imshow(rle_to_mask(sample_df[pred_col].values[i], img.shape))
                axes[0][i + 1].set_title(sample_df[title_col].values[i], fontsize=18)
                axes[1][i + 1].set_title(sample_df[title_col].values[i], fontsize=18)
                axes[0][i + 1].axis('off')
                axes[1][i + 1].axis('off')
    else:
        for sample in image_ids:
            sample_df = df[df['image'] == sample]
            fig, axes = plt.subplots(5, figsize=figsize)
            img = cv.imread(images_dest_path + sample_df['image'].values[0])
            img = cv.resize(img, img_shape)
            axes[0].imshow(img)
            axes[0].set_title('Original', fontsize=16)
            axes[0].axis('off')
            for i in range(4):
                axes[i + 1].imshow(rle_to_mask(sample_df[label_col].values[i], img.shape))
                axes[i + 1].set_title(sample_df[title_col].values[i], fontsize=18)
                axes[i + 1].axis('off')


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.ctoolsoncatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def build_rles(masks, reshape=None):
    """
    aaa
    """
    width, height, depth = masks.shape
    rles = []

    for i in range(depth):
        mask = masks[:, :, i]

        if reshape:
            mask = mask.astype(np.float32)
            mask = resize(mask, reshape).astype(np.int64)

        rle = mask2rle(mask)
        rles.append(rle)

    return rles
