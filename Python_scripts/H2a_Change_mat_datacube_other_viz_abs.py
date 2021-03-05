# -*- coding: utf-8 -*-
"""
Created on Nov 18, 2020
This script:
    Opens a shapefile and reads the geocoordinates of the polygons in it
    For a selected polygon(s) transforms geocoordinates to image coordinates, 
    Opens the RADARSAT-2 images, performs pixelwise multitemporal change detection for the polygon
    Replaces the pixels outside the polygon with nan and computes the nanmean per channel (scattering mechanism)
    Creates the change matrix for this polygon(s) 
    Interpolates the change matrix to a given number of days
    Saves results if save = 1

@author: Cristian Silva
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import SAR_utilities_V3a as sar
import fiona
#import scipy 
import rasterio
from rasterio.mask import mask
from collections import Counter
  
# -*- coding: utf-8 -*-
save=0
#folder="D:\\Datasets\\Agrisar_2009\\Mosaic\\C3\\"
#y_max=1900
#x_max=3380
################################################# Read tif raster of the SAR image ##############################
# Saved after doing the preprocessing. This contains geo info to map from 
# geocoordinates to image coordinates
tiff_path = "D:\\Paper2\\Ascending_only_V1\\"
name="Stack_V1.tif" # open geotiff sas raster with rasterio package
#name="Stack_Asc_FQ02.tif" # open geotiff sas raster with rasterio package
raster = rasterio.open(tiff_path+name)
#fig,ax=plt.subplots(1)
#plt.imshow(raster.read(1))
################################################# read shp
# Read shapefile with fiona package
Polygons_in_shp=fiona.open("D:\\Paper2\\Ascending_only_V1\\AgriSAR2009.shp", "r")
#read the polygon names and save them in a list
pol_names=[]
pol_crops=[]
for poly in range(len(Polygons_in_shp)):
    pol_names.append(Polygons_in_shp[poly]['properties']['IDENT']  )  
    pol_crops.append(Polygons_in_shp[poly]['properties']['CROP_TYPE']  )   
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
import os
imags=[]
for file in os.listdir(folder1): 
    if file.endswith(".img"):
        imags.append(file)
#################################### Extract the correspondent unique dates
mult_dates=[]
for img in imags: 
    mult_dates.append(img.split(".")[0].split("_")[-1])

dates=list(set(mult_dates))
dates.remove('26Jul2009') # remove faulty image
#dates.remove('08Jun2009')
#dates.remove('19Aug2009')
#dates.remove('12Aug2009')

dates1 = dates.copy()
dates1 = [w.replace('Jun', 'June') for w in dates1] # replace months name to names python understands
dates1 = [w.replace('Aug', 'August') for w in dates1]
dates1 = [w.replace('Jul', 'July') for w in dates1]
dates1 = [w.replace('Sep', 'September') for w in dates1]

#################################### sort the dates
import datetime
sorted_dates=sorted(dates1, key=lambda x: datetime.datetime.strptime(x, '%d%B%Y'))
dates2=sorted_dates.copy() # replace back months name to be able to call the files in the folder
dates2 = [w.replace('June', 'Jun') for w in dates2]
dates2 = [w.replace('August', 'Aug') for w in dates2]
dates2 = [w.replace('July', 'Jul') for w in dates2]
dates2 = [w.replace('September', 'Sep') for w in dates2]  

all_files=os.listdir(folder1)
x_max,y_max,header=sar.read_config_file_snap(folder1 +"\\"+T11_master)
#T_mult=np.zeros((len(dates2),x_max,y_max,3,3))

N=len(dates2)
# Empty array to save outputs of multitemporal change detection
R_avg=np.zeros([N,N,x_max,y_max])
G_avg=np.zeros([N,N,x_max,y_max])
B_avg=np.zeros([N,N,x_max,y_max])

datatype = 'float32'
basis="P"     # Lexicographic (L) or Pauli (P)
in_format="img" # Bin or img
# Multitemporal change detection of the cropped region  
# First row of change matrix, Open the image 1 (i) and do change detection with respecto to all other images in the stack
# Second row: Open next image (i+1) in the stack and do change detection 
# Continue for the n images in the stack

# instantiate class to perform eigendecompositions indicating that we will use a covariance matrix correspondint to difference of two images
application = "Difference change detection"
#eigen = sar.eigendecompositions(application)
ROI_size=[y_min,y_max,x_min,x_max]

# Instantiate class that creates a change matrix of a stack
CM_class=sar.change_matrix_of_stack(N, folder1, dates2, ROI_size, header,
                                datatype = 'float32', basis="P",in_format="img",application=application)
# Create the stack of change matrices
R_avg,G_avg,B_avg = CM_class.call()
"""
plot changes from first image to the rest

Because We havent removed areas not coincident, they may appear in the increase/decrease plots
"""
size = np.shape(R_avg[0,0,:,:])           
for img in range(len(dates2)-1):
    print(str(img+1))
    img_a=0 # change here the base/master image
    img_b=img+1 
    R_inc = R_avg[img_a,img_b,:,:]
    G_inc = G_avg[img_a,img_b,:,:]
    B_inc = B_avg[img_a,img_b,:,:]
    R_dec = R_avg[img_b,img_a,:,:] 
    G_dec = G_avg[img_b,img_a,:,:]
    B_dec = B_avg[img_b,img_a,:,:]
    a = np.zeros([size[0],size[1],3])
    
    fig,(ax,ax1)=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(15,9))
    a[:,:,0] = R_inc/np.nanmean(R_inc)
    a[:,:,1] = G_inc/np.nanmean(G_inc)
    a[:,:,2] = B_inc/np.nanmean(B_inc)
    b = np.zeros([size[0],size[1],3])
    b[:,:,0] = np.abs(R_dec)/np.abs(np.nanmean(R_dec))
    b[:,:,1] = np.abs(G_dec)/np.abs(np.nanmean(G_dec))
    b[:,:,2] = np.abs(B_dec)/np.abs(np.nanmean(B_dec))
    ax.imshow(a*0.3)
    ax1.imshow(b*0.3)
    ax.axis("off")
    ax1.axis("off")
    plt.tight_layout()


######################### create and fill datacube
CM_Datacube=np.zeros((x_max,y_max,N,N,3))
for i in range(len(dates2)):
    for j in range(len(dates2)):
        CM_Datacube[:,:,i,j,0]=R_avg[i,j,:,:] #B_avg[i,j,:,:]
        CM_Datacube[:,:,i,j,1]=G_avg[i,j,:,:] #R_avg[i,j,:,:]
        CM_Datacube[:,:,i,j,2]=B_avg[i,j,:,:] #G_avg[i,j,:,:]

"""
Plot some change matrices
"""    
pixes_i=[578,586,774,880,1045,1047]
pixes_j=[1119,1007,1082,1110,1012,1014]

fig,(axes)=plt.subplots(nrows=1, ncols=6,figsize=(18,6))
for i in range(6):
    CM=np.zeros([N,N,3]) # change matrix    
    CM1=np.zeros([N,N,3]) # change matrix  
    for img in range(N):
        for img_2 in range(N):
            CM[img,img_2,:]=(CM_Datacube[pixes_i[i],pixes_j[i],img,img_2,:]) # mean
            CM[img_2,img,:]=((CM_Datacube[pixes_i[i],pixes_j[i],img_2,img,:])) # mean
   
    axes[i].imshow(CM*4)
    axes[i].set_axis_off()
plt.tight_layout()


# If diagonals are not desired, enabled this to save        
location="D:\Paper2\Ascending_only_V1\\np_arrays\\Change_matrix\\V1\\"
layer_name = "CM_Datacube_viz_absV6" #
np.save(location+layer_name, CM_Datacube, allow_pickle=True, fix_imports=True)

"""
#################################################
################# Diagonals
#################################################
"""       
application = "Single image"
eigen = sar.eigendecompositions(application)
        
for img in range(N): ####### open image of date i and fix it   
    print(str(img+1))
    """
    ####################################### T11 ###############################
    """
    a=[]
    for file in os.listdir(folder1):
        if file.endswith(".img"):
            if dates2[img] in file:
                a.append(file)    
    ROI_size=[y_min,y_max,x_min,x_max]
    T=sar.array2D_of_coherency_matrices_from_stack_SNAP(folder1,basis,in_format,ROI_size,header,datatype,a)
    R, G, B = eigen.gral_eigendecomposition(T)
    R_avg[img,img,:,:] = R
    G_avg[img,img,:,:] = G
    B_avg[img,img,:,:] = B

##############################################save
CM_Datacube=np.zeros((x_max,y_max,N,N,3))
for i in range(len(dates2)):
    for j in range(len(dates2)):
        CM_Datacube[:,:,i,j,0]=R_avg[i,j,:,:]
        CM_Datacube[:,:,i,j,1]=G_avg[i,j,:,:]
        CM_Datacube[:,:,i,j,2]=B_avg[i,j,:,:]
        
fig,(axes)=plt.subplots(nrows=1, ncols=6,figsize=(18,6))
for i in range(6):
    CM=np.zeros([N,N,3]) # change matrix    
    CM1=np.zeros([N,N,3]) # change matrix  
    for img in range(N):
        for img_2 in range(N):
            CM[img,img_2,:]=(CM_Datacube[pixes_i[i],pixes_j[i],img,img_2,:]) # mean
            CM[img_2,img,:]=(np.abs(CM_Datacube[pixes_i[i],pixes_j[i],img_2,img,:])) # mean
   
    axes[i].imshow(CM*4)
    axes[i].set_axis_off()
plt.tight_layout()

location="D:\Paper2\Ascending_only_V1\\np_arrays\\Change_matrix\\V1\\"
layer_name = "CM_Datacube_viz_abs_diagV6" #
np.save(location+layer_name, CM_Datacube, allow_pickle=True, fix_imports=True)   




## Interpolate the change matrix
#CM1=np.nan_to_num(CM) # To interpolate we need to replace the nan of diagonals to zeros
#days=100 # days to interpolate
#CM_interpol=np.zeros([days,days,3])
#CM_interpol[:,:,0]=sar.interpolate_matrix(CM1[:,:,0], days)
#CM_interpol[:,:,1]=sar.interpolate_matrix(CM1[:,:,1], days)
#CM_interpol[:,:,2]=sar.interpolate_matrix(CM1[:,:,2], days)
#
#outall=""
#title="CM_"#+pol_names[k]
#sar.Plot_all_change_matrices(CM1,title,save,outall,factor=3)
#
#outall=""
#title="CM_Interpol_"#+pol_names[k]
#sar.Plot_all_change_matrices(CM_interpol,title,save,outall,factor=3)
