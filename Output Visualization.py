#!/usr/bin/env python
# coding: utf-8

# In[498]:


import nibabel as nib
import gzip
import shutil
import matplotlib.pyplot as plt
import numpy as np
import nrrd
import random
#from niwidgets import NiftiWidget
import nilearn
from nilearn.image import new_img_like, load_img
import os
import glob
import scipy.ndimage as ndimage
#import pydicom
import pydicom as dicom
from skimage import exposure, measure, morphology
import cv2
from PIL import Image,ImageOps


# In[50]:


#!pip install pillow


# In[2]:


angle = -270  # in degrees


# In[445]:


img_run_600 = nib.load('C:/Users/kmn14/Documents/kmn14/Thesis/Initial run/Task600_PDAC (Full run cascade)_603/predictions-600-run2/predicted labels/Epancreas_208.nii.gz').get_fdata()
img_run_600_rot = ndimage.rotate(img_run_600, angle, reshape=True)


# In[446]:


img_run_600.shape


# In[447]:


img_run_602 = nib.load('C:/Users/kmn14/Documents/kmn14/Thesis/Initial run/Task602_PDAC(excl) - Full run cascade_605/predictions-602-run1/predicted labels/Epancreas_208.nii.gz').get_fdata()
img_run_602_rot = ndimage.rotate(img_run_602, angle, reshape=True)


# In[448]:


img_original = nib.load('C:/Users/kmn14/Documents/kmn14/Thesis/Initial run/Task600_PDAC (Full run cascade)_603/imagesTs/oriented-images/Epancreas_208_0000.nii.gz').get_fdata()
img_original_rot = ndimage.rotate(img_original, angle, reshape=True)


# In[7]:


def convertNsave(arr,file_dir, index=0):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put 
    the name of each slice while using a for loop to convert all the slices
    """
    
    dicom_file = dicom.dcmread('C:/Users/kmn14/Documents/kmn14/Thesis/manifest-1612455305980/CPTAC-PDA/C3L-00017/11-22-1999-NA-XR CHEST 2 VIEWS AP OR PALAT-41732/3618.000000-Chest-56497/1-1.dcm')
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()
    dicom_file.save_as(os.path.join(file_dir, f'slice{index}.dcm'))


# In[8]:


out_dir = 'C:/Users/kmn14/Documents/kmn14/Thesis/Final report/original image'
number_slices = img_original.shape[2]
for slice_ in range(number_slices):
        convertNsave(img_original[:,:,slice_], out_dir, slice_)


# In[17]:


out_dir = 'C:/Users/kmn14/Documents/kmn14/Thesis/Final report/run-600'
number_slices = img_run_600_rot.shape[2]
for slice_ in range(number_slices):
        convertNsave(img_run_600_rot[:,:,slice_], out_dir, slice_)


# In[18]:


out_dir = 'C:/Users/kmn14/Documents/kmn14/Thesis/Final report/run-602'
number_slices = img_run_602_rot.shape[2]
for slice_ in range(number_slices):
        convertNsave(img_run_602_rot[:,:,slice_], out_dir, slice_)


# In[297]:


ds_org=dicom.dcmread('C:/Users/kmn14/Documents/kmn14/Thesis/Final report/original image/slice69.dcm')
dcm_sample_org=ds_org.pixel_array
dcm_sample_org=exposure.equalize_adapthist(dcm_sample_org)
# #cv2.imshow('sample image dicom',dcm_sample)
dcm_sample_org_rot = ndimage.rotate(dcm_sample_org, angle, reshape=True)

ds_600 = dicom.dcmread('C:/Users/kmn14/Documents/kmn14/Thesis/Final report/run-600/slice69.dcm')
dcm_sample_600=ds_600.pixel_array
dcm_sample_600=exposure.equalize_adapthist(dcm_sample_600)

ds_602 = dicom.dcmread('C:/Users/kmn14/Documents/kmn14/Thesis/Final report/run-602/slice69.dcm')
dcm_sample_602=ds_602.pixel_array
dcm_sample_602=exposure.equalize_adapthist(dcm_sample_602)


# In[677]:


n_slice = 70
dcm_sample_org = img_original[:,:,n_slice]
dcm_sample_600 = img_run_600[:,:,n_slice]
dcm_sample_602 = img_run_602[:,:,n_slice]


# In[678]:


cropped_org = dcm_sample_org[200:300,230:330]
cropped_org_unchanges = dcm_sample_org[200:300,230:330]
cropped_600 = dcm_sample_600[200:300,230:330]
cropped_602 = dcm_sample_602[200:300,230:330]


# In[679]:


print(dcm_sample_602.shape)
print(np.unique(cropped_600))
print(cropped_600.shape)


# In[680]:


RGB_600 = cv2.merge([cropped_600,cropped_600,cropped_600])
for rows_index in range(cropped_600.shape[0]):
    for columns_index in range(cropped_600.shape[1]):
        if ((RGB_600[rows_index] [columns_index] == 20).all()):
            RGB_600[rows_index] [columns_index] = np.array([0,194,113])
        elif ((RGB_600[rows_index] [columns_index] == 8).all()):
            RGB_600[rows_index] [columns_index] = np.array([231,0,206])
        elif ((RGB_600[rows_index] [columns_index] == 4).all()):
            RGB_600[rows_index] [columns_index] = np.array([251,159,255])
        elif ((RGB_600[rows_index] [columns_index] == 6).all()):
            RGB_600[rows_index] [columns_index] = np.array([0,147,0])
        elif ((RGB_600[rows_index] [columns_index] == 10).all()):
            RGB_600[rows_index] [columns_index] = np.array([157,0,0])
        elif ((RGB_600[rows_index] [columns_index] == 12).all()):
            RGB_600[rows_index] [columns_index] = np.array([157,0,0])
        elif ((RGB_600[rows_index] [columns_index] == 14).all()):
            RGB_600[rows_index] [columns_index] = np.array([157,0,0])
        elif ((RGB_600[rows_index] [columns_index] == 15).all()):
            RGB_600[rows_index] [columns_index] = np.array([0,0,255])
        elif ((RGB_600[rows_index] [columns_index] == 16).all()):
            RGB_600[rows_index] [columns_index] = np.array([0,0,255])
        elif ((RGB_600[rows_index] [columns_index] == 17).all()):
            RGB_600[rows_index] [columns_index] = np.array([0,0,255])
        elif ((RGB_600[rows_index] [columns_index] == 0).all()):
            RGB_600[rows_index] [columns_index] = np.array([255,255,255])
#         elif (((RGB_600[rows_index] [columns_index] >= 0.9).all()) and ((RGB_600[rows_index] [columns_index] <= 1.0).all())):
#             RGB_600[rows_index] [columns_index] = np.array([0,194,113])
RGB_600 = RGB_600.astype(int)


# In[681]:


cropped_org = cv2.merge([cropped_org,cropped_org,cropped_org])


# In[682]:


RGB_602 = cv2.merge([cropped_602,cropped_602,cropped_602])
for rows_index in range(cropped_602.shape[0]):
    for columns_index in range(cropped_602.shape[1]):
        if ((RGB_602[rows_index] [columns_index] == 13).all()):
            RGB_602[rows_index] [columns_index] = np.array([0,194,113])
        elif ((RGB_602[rows_index] [columns_index] == 1).all()):
            RGB_602[rows_index] [columns_index] = np.array([231,0,206])
        elif ((RGB_602[rows_index] [columns_index] == 3).all()):
            RGB_602[rows_index] [columns_index] = np.array([157,0,0])
        elif ((RGB_602[rows_index] [columns_index] == 5).all()):
            RGB_602[rows_index] [columns_index] = np.array([157,0,0])
        elif ((RGB_602[rows_index] [columns_index] == 7).all()):
            RGB_602[rows_index] [columns_index] = np.array([157,0,0])
        elif ((RGB_602[rows_index] [columns_index] == 8).all()):
            RGB_602[rows_index] [columns_index] = np.array([0,0,255])
        elif ((RGB_602[rows_index] [columns_index] == 9).all()):
            RGB_602[rows_index] [columns_index] = np.array([0,0,255])
        elif ((RGB_602[rows_index] [columns_index] == 10).all()):
            RGB_602[rows_index] [columns_index] = np.array([0,0,255])
        elif ((RGB_602[rows_index] [columns_index] == 0).all()):
            RGB_602[rows_index] [columns_index] = np.array([255,255,255])
#         elif (((RGB_600[rows_index] [columns_index] >= 0.9).all()) and ((RGB_600[rows_index] [columns_index] <= 1.0).all())):
#             RGB_600[rows_index] [columns_index] = np.array([0,194,113])
RGB_602 = RGB_602.astype(int)


# In[683]:


# RGB_600 = cv2.merge([cropped_600,cropped_600,cropped_600])
# RGB_600.shape
# for rows_index in range(79):
#     for columns_index in range(118):
#         if (((RGB_600[rows_index] [columns_index] >= 0.32).all()) and ((RGB_600[rows_index] [columns_index] <= 0.4).all())):
#             RGB_600[rows_index] [columns_index] = np.array([231,0,206])
#         elif (((RGB_600[rows_index] [columns_index] >= 0.9).all()) and ((RGB_600[rows_index] [columns_index] <= 1.0).all())):
#             RGB_600[rows_index] [columns_index] = np.array([0,194,113])

# RGB_600 = RGB_600.astype(int)


# In[684]:


# RGB_602 = cv2.merge([cropped_602,cropped_602,cropped_602])
# RGB_602.shape
# for rows_index in range(79):
#     for columns_index in range(118):
#         if (((RGB_602[rows_index] [columns_index] >= 0.32).all()) and ((RGB_602[rows_index] [columns_index] <= 0.4).all())):
#             RGB_602[rows_index] [columns_index] = np.array([231,0,206])
#         elif (((RGB_602[rows_index] [columns_index] >= 0.9).all()) and ((RGB_602[rows_index] [columns_index] <= 1.0).all())):
#             RGB_602[rows_index] [columns_index] = np.array([0,194,113])

# RGB_602 = RGB_602.astype(int)


# In[685]:


#n_slice = 70
plt.figure(figsize=(12,8))
plt.subplot(231)
plt.imshow(cropped_org_unchanges, cmap='gray')
plt.title('Original Image')
#cropped_org_unchanges = cropped_org_unchanges.astype(np.uint8)
plt.imsave('C:/Users/kmn14/Documents/kmn14/Thesis/Final report/original_img.png',cropped_org_unchanges,cmap='gray')

plt.subplot(232)
plt.imshow(RGB_600)
plt.title('Run 600')
RGB_600 = RGB_600.astype(np.uint8)
plt.imsave('C:/Users/kmn14/Documents/kmn14/Thesis/Final report/RGB_600.png',RGB_600)

plt.subplot(233)
plt.imshow(RGB_602)
plt.title('Run 602')
RGB_602 = RGB_602.astype(np.uint8)
plt.imsave('C:/Users/kmn14/Documents/kmn14/Thesis/Final report/RGB_602.png',RGB_602)


# In[690]:


plt.imshow(cropped_org_unchanges, cmap='gray')


# In[686]:


# n_slice = 70
# plt.figure(figsize=(12,8))
# plt.subplot(231)
# plt.imshow(img_original_rot[:,:,n_slice], cmap='gray')
# plt.title('Original Image')

# plt.subplot(232)
# plt.imshow(img_run_600_rot[:,:,n_slice], cmap='gray')
# plt.title('Run 600')

# plt.subplot(233)
# plt.imshow(img_run_602_rot[:,:,n_slice], cmap='gray')
# plt.title('Run 602')

# # plt.subplot(234)
# # plt.imshow(img_actual[:,:,n_slice], cmap='gray')
# # plt.title('Actual Image')


# In[687]:


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


# In[688]:


original_img = Image.open("C:/Users/kmn14/Documents/kmn14/Thesis/Final report/original_img.png")
original_img_gray = ImageOps.grayscale(original_img)
img = Image.open('C:/Users/kmn14/Documents/kmn14/Thesis/Final report/RGB_600.png')
img = img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        if item[0] > 252:
            newData.append((0, 0, 0, 255))
        else:
            newData.append(item)
            


img.putdata(newData)
img.save("C:/Users/kmn14/Documents/kmn14/Thesis/Final report/RGB_600_trans.png", "PNG")

run_600_img_trans = Image.open("C:/Users/kmn14/Documents/kmn14/Thesis/Final report/RGB_600_trans.png")
#background = overlay_transparent(original_img,run_600_img_trans,0,0)
original_img.paste(run_600_img_trans, (0, 0), run_600_img_trans)
plt.imshow(original_img)


# In[689]:


original_img = Image.open("C:/Users/kmn14/Documents/kmn14/Thesis/Final report/original_img.png")
original_img_gray = ImageOps.grayscale(original_img)
img = Image.open('C:/Users/kmn14/Documents/kmn14/Thesis/Final report/RGB_602.png')
img = img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        if item[0] > 252:
            newData.append((0, 0, 0, 255))
        else:
            newData.append(item)
            


img.putdata(newData)
img.save("C:/Users/kmn14/Documents/kmn14/Thesis/Final report/RGB_602_trans.png", "PNG")

run_602_img_trans = Image.open("C:/Users/kmn14/Documents/kmn14/Thesis/Final report/RGB_602_trans.png")
#background = overlay_transparent(original_img,run_600_img_trans,0,0)
original_img.paste(run_602_img_trans, (0, 0), run_602_img_trans)
plt.imshow(original_img)


# In[537]:


np.unique(background)


# In[ ]:





# In[ ]:




