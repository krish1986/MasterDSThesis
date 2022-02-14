#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import shutil
from collections import OrderedDict
import json
import numpy as np
import glob


# In[22]:


#Constants

OLD_TRAIN_FOLDER = 'E:/users/KMN14/Thesis/Task07_Pancreas/Task07_Pancreas_mod/train'
NEW_TRAIN_FOLDER = 'E:/users/KMN14/Thesis/Task07_Pancreas/Task07_Pancreas_mod/new_train'
#VAL_FOLDER = 'E:/users/KMN14/Thesis/Task07_Pancreas/Task07_Pancreas_mod/val' 
JSON_FOLDER = 'E:/users/KMN14/Thesis/Task07_Pancreas/Task07_Pancreas_mod/'
TASK_NAME = 'Pancreas'
DESCRIPTION = 'Pancreas and cancer segmentation'
REFRENCE = 'Memorial Sloan Kettering Cancer Center'
LICENSE = 'CC-BY-SA 4.0'
RELEASE = '0.0'
TENSOR_IMAGE_SIZE = '3D'


# In[24]:


os.chdir(TRAIN_FOLDER)
for infile in glob.glob(os.path.join( '*.nii.gz')):
    old_file_name = TRAIN_FOLDER + '/' + infile
    new_name = infile.split('.')[0] + '_' + '0000'+'.nii.gz'
    new_file_name = NEW_TRAIN_FOLDER + '/' + new_name
    os.rename(old_file_name, new_file_name)


# In[28]:


json_file_exist = False
overwrite_json_file = False

if os.path.exists(os.path.join(JSON_FOLDER,'dataset.json')):
    print('dataset.json already exist!')
    json_file_exist = True

if json_file_exist==False or overwrite_json_file:

    json_dict = OrderedDict()
    json_dict['name'] = TASK_NAME
    json_dict['description'] = DESCRIPTION
    json_dict['reference'] = REFRENCE
    json_dict['licence'] = LICENSE
    json_dict['release'] = RELEASE
    json_dict['tensorImageSize'] = TENSOR_IMAGE_SIZE

    #you may mention more than one modality
    json_dict['modality'] = {
        "0": "CT"
    }
    #labels+1 should be mentioned for all the labels in the dataset
    json_dict['labels'] = {
        "0": "background",
        "1": "kidney_r",
        "2": "kidney_l",
        "3": "adrenal_r",
        "4": "adrenal_l",
        "5": "spleen",
        "6": "gallbladder",
        "7": "pancreas",
        "8": "duodenum",
        "9": "aorta",
        "10": "tc",
        "11": "ha",
        "12": "sa",
        "13": "sma",
        "14": "pv",
        "15": "sv",
        "16": "smv",
        "17": "ivc",
        "18": "bile_duct",
        "19": "pdac",
        "20": "cysts_mass",
        "21": "stents"
    }

train_ids = os.listdir(NEW_TRAIN_FOLDER)
test_ids = []
json_dict['numTraining'] = len(train_ids)
json_dict['numTest'] = len(test_ids)

json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in train_ids]

json_dict['test'] = ["./imagesTs/%s" % (i[:i.find("_0000")]+'.nii.gz') for i in test_ids]

with open(os.path.join(JSON_FOLDER,"dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
        
if os.path.exists(os.path.join(JSON_FOLDER,'dataset.json')):
        if json_file_exist==False:
            print('dataset.json created!')
        else: 
            print('dataset.json overwritten!')


# In[27]:


train_ids


# In[ ]:




