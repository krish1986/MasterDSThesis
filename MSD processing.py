#!/usr/bin/env python
# coding: utf-8

# In[137]:


import nibabel as nib
import gzip
import shutil
import matplotlib.pyplot as plt
import numpy as np
import nrrd
import random
from niwidgets import NiftiWidget
import nilearn
from nilearn.image import new_img_like, load_img
import os


# In[138]:


#Constants

PANCREAS_LABEL= 8
TUMOR_LABEL = 13

ORG_IMG_DIR = 'E:/users/KMN14/Thesis/Task07_Pancreas/Task07_Pancreas_mod/original_image'
PRED_IMG_DIR = 'E:/users/KMN14/Thesis/nnUNet_Prediction_Results/Task501_PDAC'
SAVE_IMG_DIR = 'E:/users/KMN14/Thesis/Task07_Pancreas/Task07_Pancreas_mod/merged_image'


# In[139]:


def load_image(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img


# In[140]:


#Read original image in the folder
original_file_paths = []
os.chdir(ORG_IMG_DIR)
for infile in glob.glob(os.path.join( '*.nii.gz')):
    original_file_paths.append(ORG_IMG_DIR + '/' + infile )


# In[141]:


#Read predicted image in the folder
predicted_file_paths = []
os.chdir(PRED_IMG_DIR)
for infile in glob.glob(os.path.join( '*.nii.gz')):
    predicted_file_paths.append(PRED_IMG_DIR + '/' + infile )


# In[142]:


def merge_images(data_original, data_predicted):
    merged = np.zeros((data_original.shape[0], data_original.shape[1],data_original.shape[2]))
    for i in range(0, data_original.shape[0]):
        for j in range(0, data_original.shape[1]):
            for k in range(0,data_original.shape[2]):
                if data_original[i][j][k] == 1:
                    merged[i][j][k] = 8
                elif data_original[i][j][k] == 2:
                    merged[i][j][k] = 13
                elif (data_predicted[i][j][k] == 1 or data_predicted[i][j][k] == 2 or data_predicted[i][j][k] == 3 or
                      data_predicted[i][j][k] == 4 or data_predicted[i][j][k] == 5 or data_predicted[i][j][k] == 6 or
                      data_predicted[i][j][k] == 7 or data_predicted[i][j][k] == 9 or data_predicted[i][j][k] == 10 or
                      data_predicted[i][j][k] == 11 or data_predicted[i][j][k] == 12 or data_predicted[i][j][k] == 14):
                    merged[i][j][k] = data_predicted[i][j][k]
    return merged


# In[143]:


def validate_merged_images(merged, original, merged_label, original_label):
    array_1 = original[original == original_label]
    array_1[array_1 == original_label] = merged_label
    array_2 = merged[merged == merged_label]
    if (array_1==array_2).all():
        return
    else:
        print('Original and Merged Arrays doesnt match')


# In[144]:


def save_merged_images(merged, path, img):
    merged_image = nib.Nifti1Image(merged, img.affine, img.header)
    nib.save(merged_image, path)
    


# In[145]:


for i in range(0, len(original_file_paths)):
    if original_file_paths[i].split('/')[-1] == predicted_file_paths[i].split('/')[-1]:
        data_original, img_original = load_image(original_file_paths[i])
        data_predicted, img_predicted = load_image(predicted_file_paths[i])
        data_merged = merge_images(data_original,data_predicted)
        validate_merged_images(data_merged, data_original, PANCREAS_LABEL, 1) #Validation of Pancreas
        validate_merged_images(data_merged, data_original, TUMOR_LABEL, 2) #Validation of Tumor
        save_merged_images(data_merged, SAVE_IMG_DIR+'/'+original_file_paths[i].split('/')[-1],img_original)
        print('Total Number of images processed: ', i+1)
    else:
        print("Original and Predicted File names doesnt match")t
        print("Original File Name: ", original_file_paths[i].split('/')[-1])
        print("Predicted File Name: ", predicted_file_paths[i].split('/')[-1])        


# In[ ]:





# In[ ]:




