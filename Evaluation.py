#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
from sklearn.metrics import confusion_matrix, jaccard_score


# In[8]:


#set predicted and label working directories
pred_dir = 'C:/Users/kmn14/Documents/kmn14/Thesis/Initial run/Task602_PDAC(excl) - Full run cascade_605/predictions-605-run1 (in LISA 602)/prediction labels'
label_dir = 'C:/Users/kmn14/Documents/kmn14/Thesis/Initial run/Task602_PDAC(excl) - Full run cascade_605/predictions-605-run1 (in LISA 602)/original labels'
LABELS_IN_SCOPE = [13]


# In[9]:


#Ensure that the files are sorted so that the same predicted and label images are read

pred_file_names = []
label_file_names = []

voe_values = []
sensitivity_values = []
specificity_values = []

for pred_file in sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz"))):
    pred_file_names.append(pred_file)
for label_file in sorted(glob.glob(os.path.join(label_dir, "*.nii.gz"))):
    label_file_names.append(label_file)

if len(pred_file_names) == len(label_file_names):
    pass
else:
    print("Lenth of predicted image and original label images doesnt match")
    print("Length of predicted image ", len(pred_file_names))
    print("Length of original image ", len(label_file_names))
    sys.exit(1)
w= 23
h = len(pred_file_names)

dsc_values = [['' for x in range(w)] for y in range(h)]

for i in range(0, len(pred_file_names)):
    #Validate if the pred file names and the label file names match?
    if (pred_file_names[i].split('/') [-2]) == (label_file_names[i].split('/') [-2]):
        pass
    else:
        print("File names of predicted and label does not match")
        print("File name of predicted image ", pred_file_names[i].split('/') [-2])
        print("File name of original image ", label_file_names[i].split('/') [-2])
        sys.exit(1)
    #print(pred_file_names[i])
    #print(label_file_names[i])
    #Read the predicted file
    nii_img = nib.load(pred_file_names[i])
    pred_array = nii_img.get_fdata()

    #Read the original label file
    nii_label_img = nib.load(label_file_names[i])
    label_array = nii_label_img.get_fdata()


    def select_labels_for_dsc(array1, array2, label_value):
        array1 = (array1 == label_value)
        array2 = (array2 == label_value)
        return array1, array2
    
    def determine_dice_coefficient(array1, array2):
        y_pred = array1.flatten()
        y_actual = array2.flatten()
        intersection = np.sum(y_actual * y_pred)
        smooth = 0.0001
        return (2. * intersection + smooth) / (np.sum(y_actual) + np.sum(y_pred) + smooth)
        #select the correct label values which you are interested for evaluation
        
    def determine_voe(array1, array2):
        y_actual = array2.ravel()
        y_pred = array1.ravel()

        JSC = jaccard_score(y_actual, y_pred)
        return (1 - JSC) * 100
    
    for j in range(0, len(LABELS_IN_SCOPE)):
        pred_sel_array, label_sel_array = select_labels_for_dsc(pred_array, label_array, LABELS_IN_SCOPE[j])
        
        #Determine the Dice Coefficient
        #dsc_coef = determine_dice_coefficient(pred_sel_array, label_sel_array)
        #print(pred_file_names[i])
        #print(dsc_coef)
        #dsc_values[i][j] = dsc_coef
        voe_coef = determine_voe(pred_sel_array, label_sel_array)
        print(pred_file_names[i])
        print(voe_coef)
#print(dsc_values)


# In[10]:


# import tensorflow as tf
# from keras import backend as K
# pred_sel_array = nib.load('C:/Users/kmn14/Documents/kmn14/Thesis/Initial run/Task600_PDAC (Full run cascade)_603/predictions-600-run1/Predicted Labels/pancreas_410.nii.gz').get_fdata()
# label_sel_array = nib.load('C:/Users/kmn14/Documents/kmn14/Thesis/Initial run/Task600_PDAC (Full run cascade)_603/predictions-600-run1/Original Labels/pancreas_410.nii.gz').get_fdata()
# Conversion_dictionary = {1: 0, 2: 0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 21:0, 22:0, 23:0}
# for k, v in Conversion_dictionary.items(): pred_sel_array[pred_sel_array==k] = v
# for k, v in Conversion_dictionary.items(): label_sel_array[label_sel_array==k] = v
# array1 = K.flatten(label_sel_array)
# array2 = K.flatten(pred_sel_array)
# # array1 = pred_sel_array[pred_sel_array == 20]
# # #array2 = (label_sel_array = 20)
# # array2 = label_sel_array[label_sel_array == 20]
# smooth = 0.000001
# intersection = K.sum(array1 * array2)
# union = K.sum(array1) + K.sum(array2)
# dice = K.mean((2. * intersection + smooth)/(union + smooth))


# In[98]:


def Average(lst):
        return sum(lst) / len(lst)
def Extract(lst,item_number):
    return [item[item_number] for item in lst]


# In[1]:


# print("Mean Dice Score value, kidney right: ", Average(Extract(dsc_values, 0)))
# print("Mean Dice Score value, kidney left: ", Average(Extract(dsc_values, 1)))
# print("Mean Dice Score value, Adrenal right: ", Average(Extract(dsc_values, 2)))
# print("Mean Dice Score value, Adrenal left: ", Average(Extract(dsc_values, 3)))
# print("Mean Dice Score value, Spleen: ", Average(Extract(dsc_values, 4)))
# print("Mean Dice Score value, Liver: ", Average(Extract(dsc_values, 5)))
# print("Mean Dice Score value, Gallbladder: ", Average(Extract(dsc_values, 6)))
# print("Mean Dice Score value, Pancreas: ", Average(Extract(dsc_values, 7)))
# print("Mean Dice Score value, Duodenum: ", Average(Extract(dsc_values, 8)))
# print("Mean Dice Score value, Aorta: ", Average(Extract(dsc_values, 9)))
# print("Mean Dice Score value, TC: ", Average(Extract(dsc_values, 10)))
# print("Mean Dice Score value, HA: ", Average(Extract(dsc_values, 11)))
# print("Mean Dice Score value, SA: ", Average(Extract(dsc_values, 12)))
# print("Mean Dice Score value, SMA: ", Average(Extract(dsc_values, 13)))
# print("Mean Dice Score value, IVC: ", Average(Extract(dsc_values, 14)))
# print("Mean Dice Score value, PV: ", Average(Extract(dsc_values, 15)))
# print("Mean Dice Score value, SV: ", Average(Extract(dsc_values, 16)))
# print("Mean Dice Score value, SMV: ", Average(Extract(dsc_values, 17)))
# print("Mean Dice Score value, Bile duct: ", Average(Extract(dsc_values, 18)))
#print("Mean Dice Score value, PDAC: ", Average(Extract(dsc_values, 0)))
#print("Mean Dice Score value, kidney Cysts: ", Average(Extract(dsc_values, 20)))
#print("Mean Dice Score value, Stent: ", Average(Extract(dsc_values, 21)))
# print("Mean Dice Score value, Pancreatic Cysts: ", Average(Extract(dsc_values, 22)))


# In[ ]:




