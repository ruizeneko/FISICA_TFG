from data_load import Constants, DataGenerator
import cv2 as cv
import pandas as pd
import segmentation_models as sm
import albumentations as albu
import os
import shutil
import tools
import sklearn.model_selection
import numpy as np
import copy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from OneCycleScheduler import OneCycleScheduler
import tta_wrapper
from EDA import identification as idt
from EDA import segmentation as sgt

# First, some EDA
idt.various_counter_img()
sgt.multiple_seg_map()
# Then, we load the DF
train = pd.read_csv(Constants.PROJECT_PATH + "/" + "train.csv")

# We get images shape
img = cv.imread(Constants.PROJECT_PATH + '/train_images/' + '04df149.jpg', -1)
HEIGHT = img.shape[0]
WIDTH = img.shape[1]
DIMENSIONS = (HEIGHT, WIDTH)

# We add a column with image name and label
train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])

# Create one column for each mask
train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'],
                          aggfunc=np.min).reset_index()
train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']
corrected_df = train_df

# We create a copy from our already loaded and preprocessed DT
DF = copy.deepcopy(corrected_df)

# Why deep copy instead of assigment operator ?
# We will shuffle the list -->  https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/

# We split into train and validation sets, adding an additional column to distinguish them
X_train, X_val = sklearn.model_selection.train_test_split(DF, test_size=0.2, random_state=1997)

# We add a new column in both sets to check if it belongs to train or validation sets
X_train.insert(0, 'Set', ['Train' for _ in range(len(X_train))], True)
X_train.reset_index()
random_select_train = np.random.choice(X_train['image'].values, 50)

X_val.insert(0, 'Set', ['Val' for _ in range(len(X_val))], True)
X_val.reset_index()
random_select_val = np.random.choice(X_val['image'].values, 50)

size_train = len(X_train)
size_val = len(X_val)

print('# train examples {}'.format(size_train))
print('# val examples {}'.format(size_val))

# We create a distinct folders for both train and validation. Since there are not too much images, it will speed up
# training, because instead of finding the image from the folder and check if its training or not, it will pick up one
# randomly
train_images_path = '/kaggle/working/train_images_dict/'
validation_images_path = '/kaggle/working/val_images_dict/'

if not os.path.exists(validation_images_path):
    os.makedirs(validation_images_path)

if not tools.check_if_exist(random_select_val, 'val_images_dict'):
    if os.path.exists(validation_images_path):
        shutil.rmtree(validation_images_path)

    os.makedirs(validation_images_path)
    try:
        tools.move_data(X_val, validation_images_path)
    except IndexError:
        shutil.rmtree(validation_images_path)

# Since directories should be created only once
if not os.path.exists(train_images_path):
    os.makedirs(train_images_path)

if not tools.check_if_exist(random_select_train, 'train_images_dict'):
    if os.path.exists(train_images_path):
        shutil.rmtree(train_images_path)

    os.makedirs(train_images_path)
    try:
        tools.move_data(X_train, train_images_path)
        print('moved and created')
    except IndexError:
        shutil.rmtree(train_images_path)

# We set the hyperparameters of our training
backbones = ['resnet34', 'resnet152', 'densenet121', 'efficientnetb2']
BACKBONE = backbones[0]
BATCH_SIZE = 14
EPOCHS = 15
LEARNING_RATE = 1e-4
HEIGHT = 384
WIDTH = 480
CHANNELS = 3
N_CLASSES = 4
ES_PATIENCE = 5
RLROP_PATIENCE = 3
DECAY_DROP = 0.5
model_path = 'uNet_%s_%sx%s.h5' % (BACKBONE, HEIGHT, WIDTH)

# We add augmentation and preprocessing
preprocessing = sm.get_preprocessing(BACKBONE)

augmentation = albu.Compose([albu.HorizontalFlip(),
                             albu.VerticalFlip(),
                             albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1),
                             ])

# we load the generators
train_generator = DataGenerator.DataGenerator(
    directory=train_images_path,
    dataframe=X_train,
    target_df=train,
    batch_size=BATCH_SIZE,
    target_size=(HEIGHT, WIDTH),
    n_channels=CHANNELS,
    n_classes=N_CLASSES,
    augmentation=augmentation,
    preprocessing=preprocessing)

valid_generator = DataGenerator.DataGenerator(
    directory=validation_images_path,
    dataframe=X_val,
    target_df=train,
    batch_size=BATCH_SIZE,
    target_size=(HEIGHT, WIDTH),
    n_channels=CHANNELS,
    n_classes=N_CLASSES,
    preprocessing=preprocessing)

# We load the model
model = sm.Unet(backbone_name=BACKBONE,
                encoder_weights='imagenet',
                classes=4,
                activation='sigmoid',
                input_shape=(HEIGHT, WIDTH, 3))

# We load the callbacks and the metrics we will use
checkpoint = ModelCheckpoint(model_path, mode='min', save_best_only=True, save_weights_only=True)
es = EarlyStopping(mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)

onecycle = OneCycleScheduler(len(X_train) // BATCH_SIZE * EPOCHS, max_rate=0.05)

metric_list = [tools.dice_coef, sm.metrics.iou_score]
callback_list = [checkpoint, es, onecycle]

# the optimizers, in this specific case SGD
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=True)
model.compile(optimizer, loss=sm.losses.bce_dice_loss, metrics=metric_list)
model.summary()

STEP_SIZE_TRAIN = len(X_train) // BATCH_SIZE
STEP_SIZE_VALID = len(X_val) // BATCH_SIZE

# We train our model. CAUTION: If you do not have GPU, this may take a lot of time
history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    callbacks=callback_list,
    epochs=EPOCHS,
    verbose=1)

# We plot the values get in each epoch
tools.plot_metrics(history.history, metric_list=['loss', 'dice_coef', 'iou_score'])

# We define the threshold to delete a mask (if its too small, it may be noise). The size are picked after try-fail.
# Sugar is bigger because is bigger too in satellite images.
class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
best_thresholds = [.5, .5, .5, .35]
best_masks = [25000, 20000, 22500, 15000]

for index, name in enumerate(class_names):
    print('%s treshold=%.2f mask size=%d' % (name, best_thresholds[index], best_masks[index]))

# We calculate the accuracy after applying the previous filter
train_metrics = tools.get_metrics(model, train, X_train, train_images_path, best_thresholds, best_masks,
                                  preprocessing=preprocessing, set_name='Train')
print(train_metrics)
validation_metrics = tools.get_metrics(model, train, X_val, validation_images_path, best_thresholds, best_masks,
                                       preprocessing=preprocessing, set_name='Validation')
print(validation_metrics)

# Test time augmentation. It has been taken from https://github.com/qubvel/tta_wrapper
# and adapted too TensorFlow 2.
model = tta_wrapper.tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10))

# We repeat the same proccess done for validation and for training for test set.
submission = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')
submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
test = pd.DataFrame(submission['image'].unique(), columns=['image'])

random_select_test = np.random.choice(test['image'].values, 30)

test_images_path = '/kaggle/working/test_images_dict/'
if not os.path.exists(test_images_path):
    os.makedirs(test_images_path)

if not tools.check_if_exist(random_select_test, 'test_images_dict'):
    if os.path.exists(test_images_path):
        shutil.rmtree(test_images_path)

    os.makedirs(test_images_path)
    try:
        tools.move_data(test, test_images_path, test_train='test')
        print('created')
    except IndexError:
        shutil.rmtree(test_images_path)

test_df = []

# We do the predictions for test set
for i in range(0, test.shape[0], 300):
    batch_idx = list(range(i, min(test.shape[0], i + 300)))
    batch_set = test[batch_idx[0]: batch_idx[-1] + 1]

    # We load test set
    test_generator = DataGenerator.DataGenerator(
        directory=test_images_path,
        dataframe=batch_set,
        target_df=submission,
        batch_size=1,
        target_size=(HEIGHT, WIDTH),
        n_channels=CHANNELS,
        n_classes=N_CLASSES,
        preprocessing=preprocessing,
        mode='predict',
        shuffle=False)

    model.run_eagerly = True
    predictions = model.predict_generator(test_generator)

    for index, b in enumerate(batch_idx):
        filename = test['image'].iloc[b]
        image_df = submission[submission['image'] == filename].copy()
        pred_masks = predictions[index,].round().astype(int)
        pred_rles = tools.build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels'] = pred_rles

        # Post procecssing
        pred_masks_post = predictions[index,].astype('float32')
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[..., class_index]
            pred_mask = tools.post_process(pred_mask, threshold=best_thresholds[class_index],
                                           min_size=best_masks[class_index])
            pred_masks_post[..., class_index] = pred_mask

        pred_rles_post = tools.build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_post'] = pred_rles_post
        ###

        test_df.append(image_df)

sub_df = pd.concat(test_df)

# Choose 3 samples at random from validation test. First we show the correct masks and afterwards the prediction
images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
inspect_set = train[train['image'].isin(images_to_inspect)].copy()
inspect_set_temp = []

inspect_generator = DataGenerator.DataGenerator(
    directory=validation_images_path,
    dataframe=inspect_set,
    target_df=train,
    batch_size=1,
    target_size=(HEIGHT, WIDTH),
    n_channels=CHANNELS,
    n_classes=N_CLASSES,
    preprocessing=preprocessing,
    shuffle=False)

predictions = model.predict_generator(inspect_generator)

for index, b in enumerate(range(len(predictions))):
    filename = inspect_set['image'].iloc[b]
    image_df = inspect_set[inspect_set['image'] == filename].copy()
    pred_masks = predictions[index,].round().astype(int)
    pred_rles = tools.build_rles(pred_masks, reshape=(350, 525))
    image_df['EncodedPixels_pred'] = pred_rles

    # Post procecssing
    pred_masks_post = predictions[index,].astype('float32')
    for class_index in range(N_CLASSES):
        pred_mask = pred_masks_post[..., class_index]
        pred_mask = tools.post_process(pred_mask, threshold=best_thresholds[class_index],
                                       min_size=best_masks[class_index])
        pred_masks_post[..., class_index] = pred_mask

    pred_rles_post = tools.build_rles(pred_masks_post, reshape=(350, 525))
    image_df['EncodedPixels_pred_post'] = pred_rles_post
    ###
    inspect_set_temp.append(image_df)

# Without postprocessing
inspect_set = pd.concat(inspect_set_temp)
tools.inspect_predictions(inspect_set, images_to_inspect, validation_images_path, pred_col='EncodedPixels_pred')

# With postprocessing
tools.inspect_predictions(inspect_set, images_to_inspect, validation_images_path, pred_col='EncodedPixels_pred_post')

# Choose 5 samples at random from test set
images_to_inspect_test = np.random.choice(sub_df['image'].unique(), 4, replace=False)

# Without postprocessing
tools.inspect_predictions(sub_df, images_to_inspect_test, test_images_path)

# With postprocessing
tools.inspect_predictions(sub_df, images_to_inspect_test, test_images_path, label_col='EncodedPixels_post')

# We fill the .csv, one without postprocessing and another with
submission_df_post = sub_df[['Image_Label', 'EncodedPixels_post']]
submission_df_post.columns = ['Image_Label', 'EncodedPixels']
submission_df_post.to_csv('submission_post.csv', index=False)
print(submission_df_post.head())

submission_df = sub_df[['Image_Label', 'EncodedPixels']]
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())
