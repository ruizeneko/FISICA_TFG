# TFG (Bachelor Thesis) of Eneko Ruiz 
### Introduction
This is the Bachelor Thesis presented in the course 2019/2020 by Eneko Ruiz to finish him Bachelor of Science in Physics. It was presented and defended virtually due to restrictions imposed by the ongoing COVID-19 pandemic. Here briefly the main points of the coding part of the work will be presented. It was mainly runned on Kaggle, to take advantage of free GPU usage available there. It is available here: https://www.kaggle.com/ruizeneko/eneko-tfg

All the packages that are used and are not coded are imported but https://github.com/qubvel/tta_wrapper. It has been downloaded and changes to run in TensorFlow 2. Very minor changes were made.

However, it could be executed in our own computer too (as long as we have both enough RAM and patience). If you would like to run this in your computer, please, follow these steps: 

1. Install Python 3.7 (or the available version when you are reading this) from https://www.python.org/downloads/
2. Install Conda environment https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
3. Import the environment used by me with ```conda env create --file tfg_conda.yml``` in your cmd or console. tfg_conda.yml is available in this repository. If you need more info, please follow either https://github.com/ruizeneko/fisica_TFG/edit/master/README.md or https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf.

Installing the Conda environment is extremly important, because some packages (segmentation_models for instance) do not work with all version of TensorFlow. Next, we will have to load the data from Kaggle : https://www.kaggle.com/c/understanding_cloud_organization/data. Please, unzip the data in the same folder of these code and do not change any name, neither of folder nor of .csv files.

###How-to-use
If you want to execute the code, you just have to run the train.py files (after unzipping the data). It will shown the segmentations maps and boxes of some randomly picked images and will start the training proccess. After it, it will create the .csv from the test set and show some randomly picked images and their masks. However, please, unless you have a GPU, use the Kaggle notebook from above. If you have a Google account, then you have a Kaggle's one too.
