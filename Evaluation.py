import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
from sklearn.metrics import confusion_matrix, jaccard_score


# data = np.load('/scratch/kmuthumanavasudevan/nnUNet/nnunet/nnUNet_Prediction_Results/Task501_PDAC/Pancreas_055.npz')
# lst = data.files
# for item in lst:
#     print(item)
#     #print(data[item][1])
#     print(np.shape(data[item][1]))

#set predicted and label working directories
pred_dir = '/scratch/kmuthumanavasudevan/nnUNet/nnunet/nnUNet_Prediction_Results/Task501_PDAC/'
label_dir = '/scratch/kmuthumanavasudevan/Label dir/'
DSC_LABEL = 5
LABELS_IN_SCOPE = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

#Ensure that the files are sorted so that the same predicted and label images are read

pred_file_names = []
label_file_names = []
dsc_values = []
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

for i in range(0, len(pred_file_names)):
    #Validate if the pred file names and the label file names match?
    if (pred_file_names[i].split('/') [-1]) == (label_file_names[i].split('/') [-1]):
        pass
    else:
        print("File names of predicted and label does not match")
        print("File name of predicted image ", pred_file_names[i])
        print("File name of original image ", label_file_names[i])
        sys.exit(1)

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


    def Average(lst):
        return sum(lst) / len(lst)

    def determine_voe(array1, array2):
        y_actual = array2.ravel()
        y_pred = array1.ravel()

        JSC = jaccard_score(y_actual, y_pred)
        return (1 - JSC) * 100

    def det_conf_matrix(array1, array2):
        y_pred = array1.ravel()
        y_actual = array2.ravel()
        TN, FP, FN, TP = confusion_matrix(y_actual, y_pred).ravel()
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        return sensitivity, specificity



    #select the correct label values which you are interested for evaluation
    pred_sel_array, label_sel_array = select_labels_for_dsc(pred_array, label_array, DSC_LABEL)

    #Determine the Dice Coefficient
    dsc_coef = determine_dice_coefficient(pred_sel_array, label_sel_array)
    dsc_values.append(dsc_coef)

    #Determine the VOE
    voe_coef = determine_voe(pred_sel_array, label_sel_array)
    voe_values.append(voe_coef)

    #Determine Confusion Matrix
    sensitivity, specificity  = det_conf_matrix(pred_sel_array, label_sel_array)
    sensitivity_values.append(sensitivity)
    specificity_values.append(specificity)

print("Mean Dice Score value: ", Average(dsc_values))
print("Mean VOE value: ", Average(voe_values))
print("Mean Sensitivity value: ", Average(sensitivity_values))
print("Mean Specificity value: ", Average(specificity_values))





# #n_slice = np.random.randint(0, nii_data.shape[2])
# n_slice = 239
# plt.figure(figsize=(12,8))
# plt.subplot(231)
# plt.imshow(nii_data[:,:,n_slice], cmap='gray')
# plt.title(' Predicted Label')
#
# plt.subplot(232)
# plt.imshow(nii_data_label[:,:,n_slice], cmap='gray')
# plt.title('Original Label')
# plt.savefig('/scratch/kmuthumanavasudevan/Test images/figure.jpg')
