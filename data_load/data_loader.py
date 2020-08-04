import identification as idt
import numpy as np

DF = idt.corrected_df
LABELS = ['Fish', 'Flower', 'Gravel', 'Sugar']
TARGET_SIZE = 224, 224

# Get a list with all images name
image_list = np.array(list(map(lambda x: str(x), DF['Image'][0:-1:4])))

# Get NotNa() elements, converts to numpy boolean array, converts the list to integer and divides by image
image_labels_one_hot = np.split(np.array(DF.notna()['EncodedPixels'].to_list()).astype(int), len(DF)/4)
