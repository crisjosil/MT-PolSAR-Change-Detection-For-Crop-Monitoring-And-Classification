# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:12:44 2020
This script creates the array2D_of_coherency_matrices for every image in the stack obtained after 
pre-processing in SNAP. Using this script there is no need to manually modify the names of the files in the stack folder
It ensures to keep only the pixels that intersect in all images. Remove where there is no overlap
so that all images in stack have exact same pixels
@author: Cristian Silva
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt 
import numpy as np
import os
#os.chdir('D:\\Juanma - Agrisar 2009\\Full_data\\Python\\Temporal Change detection')
import SAR_utilities_V3a as sar
import datetime
#folder="D:\\Juanma\\Seville 2014\\FQ19W\\2014-07-23.rds2\\"
#folder="D:\\DATASETS\\cm6406_3_Frequency_Polarimetry_San_Fran\\cm6406\\C3\\"
#dx=1168   # San francisco L band
#dy=2531
#folder="D:\\Datasets\\AIRSAR_SanFrancisco\\C3\\"
#y_max=900   # San francisco test sample
#x_max=1024
#folder="D:\\Datasets\\Agrisar_2009\\Mosaic\\C3\\"
#y_max=1900
#x_max=3380       
"""
###########################################################################################################################################################
#################################################################### open image components ############################
###########################################################################################################################################################
"""
x_min=0
y_min=0
datatype = 'float32'
T11_master ="T11_mst_08Sep2009"
folder1 = "D:\\Paper2\\Ascending_only_V1\\Stack_V1.data"

#################################### Append files in folder that end with .img
imags=[]
for file in os.listdir(folder1): 
    if file.endswith(".img"):
        imags.append(file)
#################################### Extract the correspondent unique dates
mult_dates=[]
for img in imags: 
    mult_dates.append(img.split(".")[0].split("_")[-1])

dates=list(set(mult_dates))
dates.remove('26Jul2009')
#dates.remove('08Jun2009')
#dates.remove('19Aug2009')
#dates.remove('12Aug2009')

dates1 = dates.copy()
dates1 = [w.replace('Jun', 'June') for w in dates1]
dates1 = [w.replace('Aug', 'August') for w in dates1]
dates1 = [w.replace('Jul', 'July') for w in dates1]
dates1 = [w.replace('Sep', 'September') for w in dates1]

#################################### sort the dates
sorted_dates=sorted(dates1, key=lambda x: datetime.datetime.strptime(x, '%d%B%Y'))
dates2=sorted_dates.copy()
dates2 = [w.replace('June', 'Jun') for w in dates2]
dates2 = [w.replace('August', 'Aug') for w in dates2]
dates2 = [w.replace('July', 'Jul') for w in dates2]
dates2 = [w.replace('September', 'Sep') for w in dates2] 

#for k in range(len(dates2)):
#    a=[]
#    for file in os.listdir(folder1):
#        if file.endswith(".img"):
#            if dates[k] in file:
#                a.append(file)        
#    T11,T22,T33,T12,T13,T23=read_Img_components_from_stack_SNAP(basis,folder1,a)    
#    sar.visRGB_from_T(T22, T33, T11,"",factor=5)    

all_files=os.listdir(folder1)
x_max,y_max,header=sar.read_config_file_snap(folder1 +"\\"+T11_master)
T_mult=np.zeros((len(dates2),x_max,y_max,3,3),dtype=complex)
basis = "P" 

for i in range(len(dates2)):
    a=[]
    for file in os.listdir(folder1):
        if file.endswith(".img"):
            if dates[i] in file:
                a.append(file)
    print(a)            
    #folder=folders[i]
    #x_max,y_max,header=sar.read_config_file_snap(folder+"T11")
    basis="P"     # Lexicographic (L) or Pauli (P)
    in_format="img" # Bin or img
    
    #################################
    ############### Crop the image to some known size otherwise comment (disable) the lines below
    #################################
    ROI_size=[y_min,y_max,x_min,x_max]
    # Compute image size
    #da=(x_max-x_min)
    #dr=(y_max-y_min) 
    """  
    ###########################################################################################################################################################
    ###########################################################################################################################################################
    # Create the covariance matrix for each resolution cell                      ############################
    # Save the covariance matrices in a 2D array just like original image        ############################
    # If components are in Lexicographic basis, transforms to coherency matrix   ############################
    # So far only tested for .Bin products 
    ###########################################################################################################################################################
    ###########################################################################################################################################################
    """
    T_mult[i,:,:]=sar.array2D_of_coherency_matrices_from_stack_SNAP(folder1,basis,in_format,ROI_size,header,datatype,a)

for i in range(len(dates2)):
    sar.visRGB_from_T(T_mult[i,:,:,1,1], T_mult[i,:,:,2,2], T_mult[i,:,:,0,0],str(dates2[i]),factor=5)

# Only keep the pixels that intersect in all images. Remove where there is no overlap
for i in range(len(dates2)):           # for every image in the stack
    print("Evaluating image "+str(i+1)+" of "+str(len(dates2)))
    for k in range(T_mult.shape[1]):
        for l in range(T_mult.shape[2]): # evaluate the pixel i,j
            if T_mult[i,k,l,0,0]==0:     # if its zero
                T_mult[:,k,l,:,:]=0      # make that pixel equal to zero in all the rest of the stack

#T_mult=T_mult[:,340:T_mult.shape[1],0:4800,:,:] # crop
            
for i in range(len(dates2)):
    sar.visRGB_from_T(T_mult[i,:,:,1,1], T_mult[i,:,:,2,2], T_mult[i,:,:,0,0],str(dates2[i]),factor=9)

stack_name="T_stack_asc"#"T_stack_asc"
np.save("D:\Paper2\Ascending_only_V1\\np_arrays\\"+stack_name, T_mult, allow_pickle=True, fix_imports=True)

#import xarray as xr
#data = xr.DataArray(np.random.randn(2, 3), dims=("x", "y"), coords={"x": [10, 20]})
#print(data)

a=[]
i=1
for file in os.listdir(folder1):
    if file.endswith(".img"):
        if dates[i] in file:
            a.append(file)
C11,C22,C33,C12,C13,C23=sar.read_Img_components_from_stack_SNAP(basis,folder1,a)
