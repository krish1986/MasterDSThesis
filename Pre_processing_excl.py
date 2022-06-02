#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import nibabel as nib
import os
import glob
import matplotlib.pyplot as plt
import copy


# In[74]:


input_file_path ='/home/kris/Test labels/'
output_file_path = '/home/kris/Test labels_excl/'


# In[75]:


Conversion_dictionary = {1: 0, 2: 0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:1, 9:2, 10:3, 11:4, 12:5, 13:6, 14:7, 15:8, 16:9, 17:10, 18:11, 19:12, 20:13, 21:0, 22:14, 23:15}


# In[76]:


original_file_paths = []
os.chdir(input_file_path)
for infile in glob.glob(os.path.join( '*.nii.gz')):
    original_file_paths.append(input_file_path + '/' + infile )


# In[78]:


for i in range(0, len(original_file_paths)):
    img = nib.load(original_file_paths[i])
    data = img.get_fdata()
    #newArray = np.zeros((data.shape[0], data.shape[1],data.shape[2]),dtype=np.uint8)
    #newArray = copy.deepcopy(data,dtype=np.uint8)
    for k, v in Conversion_dictionary.items(): data[data==k] = v
    data_int = data.astype(int)
    merged_image = nib.Nifti1Image(data_int, img.affine, img.header)
    save_name = output_file_path+'/'+original_file_paths[i].split('/')[-1]
    nib.save(merged_image, save_name)

