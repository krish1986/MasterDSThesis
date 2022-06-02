#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import nibabel as nib
import os
import glob
import matplotlib.pyplot as plt
import copy


# In[65]:


input_file_path ='C:/Users/kmn14/Documents/kmn14/Thesis/krish scans/PP scans/Labels'
output_file_path = 'C:/Users/kmn14/Documents/kmn14/Thesis/Initial run/Task602_PDAC(excl) - Full run cascade_605/reorientation_labels_PP.txt'


# In[66]:


original_file_paths = []
os.chdir(input_file_path)
for infile in glob.glob(os.path.join( '*.nii.gz')):
    original_file_paths.append(input_file_path + '/' + infile )


# In[ ]:





# In[67]:


original_file_paths[0].split('/')[-1]


# In[68]:


for i in range(0, len(original_file_paths)):
    lines = 'fslreorient2std' + ' ' + '/home/kris/PP_labels_excl/' + original_file_paths[i].split('/')[-1] + ' ' + '/home/kris/PP_labels_excl_oriented/' + original_file_paths[i].split('/')[-1].split('.')[-3]
    with open(output_file_path, 'a') as f:
        f.write(lines)
        f.write('\n')


# In[ ]:




