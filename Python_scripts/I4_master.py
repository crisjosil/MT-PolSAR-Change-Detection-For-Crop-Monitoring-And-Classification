# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:40:38 2020
This file splits the ground truth into training, validation and test datasets, creating an
array that contains a flag for every pixel in the ground truth, indicating whether it is train\val\test

Then it takes the previously created PolSAR and change matrixe datacubes, and break them according to the
flags array to create the train/test/ validation datasets that will be used to train the classifiers
@author: Cristian Silva-Perez
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import SAR_utilities_V3a as sar
import D2_Ground_truth_summary as gr_truth
import I3_Final_dataset as final

print("Creating flags ...")
# instantiate ground truth class as an object here
ground_truth=gr_truth.split_ground_truth(split_train=0.65,split_validation=0.15)
# change default paths if needed
ground_truth.load_arrays()
ground_truth.call() # execute
ground_truth.plots_figs() # plots
ground_truth.Plot_map_with_legend
ground_truth.save_arrays(path_flags = "D:\Paper2\Ascending_only_V1\\np_arrays\\V10\\Train_test_flags_asc_Valid",
                         path_Labels_corrected = "D:\Paper2\Ascending_only_V1\\np_arrays\\V10\\Labels_corrected") # change default paths if needed
#print(ground_truth.tr_ts_df)


print("Creating train/validation/test datasets ...")
# instantiate final_datasets class as an object here
final_datasets=final.create_datasets()
#final_datasets.load_in_arrays() # change default paths if needed. Disabled since we will load the input arrays here
final_datasets.labels = ground_truth.labels_
final_datasets.tr_ts_arr = ground_truth.tr_ts_arr


# Different visualizations of the change matrix were created, therefore we can create datasets for each of this matrix representations as below.
# In case of only needing the one presented in the paper, remove the loop, indent back and make i = 0.
#CM_cubes=['CM_Datacube.npy','CM_Datacube_with_diag.npy','CM_Datacube_viz_alb.npy','CM_Datacube_with_diag_viz_alb.npy']
#name_comp=['_','_with_diag','_viz_alb','_with_diag_viz_alb']

CM_cubes=['CM_Datacube_viz_absV6.npy','CM_Datacube_viz_abs_with_diagV6.npy']
name_comp=['_','_with_diag']

print("loading PolSAR datacube ...")
path_polSAR_datacube='D:\\Paper2\\Ascending_only_V1\\np_arrays\\datacube_MT_asc.npy'
final_datasets.datacube_MT=np.load(path_polSAR_datacube,  allow_pickle=True)
save_PolSAR = True

for i in range(len(CM_cubes)):
    print("CM "+str(i+1)+' of '+str(len(CM_cubes)))
    path_CM_datacube='D:\\Paper2\\Ascending_only_V1\\np_arrays\\Change_matrix\\V1\\'+CM_cubes[i]
    var=name_comp[i]
    CM_arr_names = ["X_train"+var,"y_train"+var,"X_val"+var,"y_val"+var,"X_test"+var,"y_test"+var]
    print("loading change matrix cube ...")
    final_datasets.CM_Datacube=np.load(path_CM_datacube,  allow_pickle=True)

    final_datasets.call() # execute operations
    final_datasets.save_arrays(save_PolSAR=save_PolSAR,save_CM=True,
                        location_PolSAR="D:\\Paper2\\Ascending_only_V1\\np_arrays\\Final_datasets\\PolSAR_F\\V10\\",
                        location_CM="D:\\Paper2\\Ascending_only_V1\\np_arrays\\Final_datasets\\CM\\V10\\",
                        PolSAR_arr_names=["X_train","y_train","X_val","y_val","X_test","y_test"],
                        CM_arr_names=CM_arr_names)
    save_PolSAR = False
