# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:52:51 2020
This script creates a numpy array with the pixelwise labels to be used as targets when training an ML classifier
It also creates another layer (np array) with the pixelwise parcel name (useful later when splitting into train/test)
Procedure:
Reads geotiff raster of the master image in the SAR stack (or any image since the stack is corregistered)
Reads shp (Make sure in QGIS that the geotiff and shp are in same crs)
Creates a dictionary relating every target class with a class number, i.e.canola:1, Barley:2, ...
Creates an empty array with shape as the master image
Selects every polygon in the shp with area greater than 20Ha, gets geometry, identify pixels inside it and assign them to the class 
    according to the shapefile
@author: crs2
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt 
import numpy as np
import geopandas as gpd
import os
#os.chdir('D:\\Juanma - Agrisar 2009\\Full_data\\Python\\Temporal Change detection')
import SAR_utilities_V3a as sar
import rasterio
import fiona
from rasterio.mask import mask

################################################# Read tif raster of the SAR image ##############################
# Saved after doing the preprocessing. This contains geo info to map from 
# geocoordinates to image coordinates
tiff_path = "D:\\Paper2\\Ascending_only_V1\\"
name="Stack_V1.tif" # open geotiff sas raster with rasterio package
#name="Stack_Asc_FQ02.tif" # open geotiff sas raster with rasterio package
raster = rasterio.open(tiff_path+name)
fig,ax=plt.subplots(1)
plt.imshow(raster.read(1))
################################################# read shp
# Read shapefile with fiona package
Polygons_in_shp=fiona.open("D:\\Paper2\\Ascending_only_V1\\AgriSAR2009.shp", "r")
#read the polygon names and save them in a list
pol_names=[]
pol_crops=[]
for poly in range(len(Polygons_in_shp)):
    pol_names.append(Polygons_in_shp[poly]['properties']['IDENT']  )  
    pol_crops.append(Polygons_in_shp[poly]['properties']['CROP_TYPE']  )  
#print(pol_names)
# initializing lists 
crop_types = list(set(pol_crops))
crop_types.sort()
crop_types_code = np.arange(0,len(crop_types))
  # using dictionary comprehension to convert lists to dictionary 
crop_types_dict_code = {crop_types[i]: crop_types_code[i] for i in range(len(crop_types))} 

#save=1
#for k in range(len(pol_names)):
#print("Parcel "+str(k+1)+" of "+str(len(pol_names)))

# label an empty image with pixel targets
targets   = np.empty((raster.shape[0],raster.shape[1]))
area_array= np.empty((raster.shape[0],raster.shape[1]))
id_array  = np.empty(shape=(raster.shape[0],raster.shape[1]), dtype=object)
#targets=np.empty(T_mult[2,:,:,1,1].shape)
targets[:] = np.nan
area_array[:] = np.nan
id_array[:] = np.nan
# Type polygon to process 
outside_pol=0
pad=False
pad_=7

def pad_me(numpy_polygon_x,pad_m,fill_with=0):
    numpy_polygon_x[:,0:pad_m]=fill_with
    numpy_polygon_x[0:pad_m,:]=fill_with
    numpy_polygon_x[(numpy_polygon[:,0].shape[-1]-pad_m):numpy_polygon[:,0].shape[-1],:]=fill_with
    numpy_polygon_x[:,(numpy_polygon.shape[-1]-pad_m):numpy_polygon.shape[-1]]=fill_with
    return(numpy_polygon_x)
    
for i in range(len(pol_names)):
    print(str(i+1)+" of "+str(len(pol_names)))
    poly=i#'0  #e.g. the first polygon
    
    
    polygon_A=Polygons_in_shp[poly]["geometry"] # filter the polygon from the shp
    crop=Polygons_in_shp[poly]["properties"]['CROP_TYPE']
    area=Polygons_in_shp[poly]["properties"]['AREA_HA']
    parcel_ID=Polygons_in_shp[poly]["properties"]['IDENT']
    ################################################# Mask a polygon ###########################
    if area>20:
    # use rasterio to mask and obtain the pixels inside the polygon. 
    # Because the mask method of rasterio requires an interable object to work, we do:
        polygon_A_list=[polygon_A] 
        try:
            # The out_image variable in the next line is a np array of the polygon desired
            out_image, out_transform = mask(raster, polygon_A_list, crop=True,pad =pad)  # go to mask source code, change in line 70, pad to 5 pixels
        except:
            
            outside_pol=outside_pol+1
            print("Polygons outside the raster: "+str(outside_pol))
        else:
            ################################################ Get image coord of the corners of the polygon(subset)
            # Obtain the polygon as numpy array
            numpy_polygon=out_image[:,:,:][0]
            # affine transform of rasterio: This contains info about where the image starts, what the resolution is, etc.. 
            subset_origin=~raster.transform * (out_transform[2], out_transform[5])
            # coordinates of top-left 
            subset_originX = int(subset_origin[0])
            subset_originY = int(subset_origin[1])
            # size of the cropped subset after masking
            pixels_in_row=int(numpy_polygon.shape[1])
            pixels_in_col=int(numpy_polygon.shape[0])
            # mask out pixels outside the polygon and fill sub arrays with class(crop type), area and parcel ID
            numpy_polygon1=np.where(numpy_polygon==0,np.nan,crop_types_dict_code[Polygons_in_shp[poly]["properties"]['CROP_TYPE']])
            numpy_polygon2=np.where(numpy_polygon==0,np.nan,area)
            numpy_polygon3=np.where(numpy_polygon==0,np.nan,parcel_ID)
            numpy_polygon3=numpy_polygon3.astype(object)
    
            numpy_polygon1= pad_me(numpy_polygon1,pad_,fill_with=np.nan)
            numpy_polygon2= pad_me(numpy_polygon2,pad_,fill_with=np.nan)
            numpy_polygon3= pad_me(numpy_polygon3,pad_,fill_with=np.nan)
            
            # fill the big array with sub arrays(polygons)
            for k in range(numpy_polygon.shape[0]):
                for l in range(numpy_polygon.shape[1]):        
                    i=subset_originY+k
                    j=subset_originX+l
                    targets[i,j]=numpy_polygon1[k,l]
                    area_array[i,j]=numpy_polygon2[k,l]
                    id_array[i,j]=numpy_polygon3[k,l]
            #targets[i,j]=1

#fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
#title="Labels"
#ax1.set_title(title)
#ax1.imshow(numpy_polygon)
#plt.axis("off")
#plt.tight_layout()
#
#
#numpy_polygon3=numpy_polygon3.astype(object)
#numpy_polygon_x= pad_me(numpy_polygon3,pad_,fill_with=np.nan)
#
#fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
#title="Labels"
#ax1.set_title(title)
#ax1.imshow(numpy_polygon_x)
#plt.axis("off")
#plt.tight_layout()
        
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
title="Labels"
ax1.set_title(title)
ax1.imshow(targets)
plt.axis("off")
plt.tight_layout()

fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
title="Area"
ax1.set_title(title)
ax1.imshow(area_array)
plt.axis("off")
plt.tight_layout()

"""Enable the following line"""
T_mult=np.load("D:\Paper2\Ascending_only_V1\\np_arrays\\T_stack_asc.npy")
#T_mult=np.load('X:\\crs2\\Paper2_Agrisar\\T_Stack\\T_stack_asc.npy')

## Only keep the pixels that intersect in all images. Remove where there is no overlap
targets1=targets.copy()
area_array1=area_array.copy()
id_array1=id_array.copy()
for k in range(T_mult.shape[1]):
    for l in range(T_mult.shape[2]):
        if T_mult[0,k,l,0,0]==0:
            #T_mult[1,k,l,:,:]=0
            targets1[k,l]=np.nan
            area_array1[k,l]=np.nan
            id_array1[k,l]=np.nan
            
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
title="Hi"
ax1.set_title(title)
ax1.imshow(targets1)
plt.axis("off")
plt.tight_layout()

fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
title="Area"
ax1.set_title(title)
ax1.imshow(area_array1)
plt.axis("off")
plt.tight_layout()
#
layer_name = "Labels_asc" # "Labels_asc"
np.save("D:\Paper2\Ascending_only_V1\\np_arrays\\"+layer_name, targets1, allow_pickle=True, fix_imports=True)
layer_name = "IDs_asc" # "IDs_asc"
np.save("D:\Paper2\Ascending_only_V1\\np_arrays\\"+layer_name, id_array1, allow_pickle=True, fix_imports=True)
layer_name = "Areas_asc" # "Areas_asc"
np.save("D:\Paper2\Ascending_only_V1\\np_arrays\\"+layer_name, area_array1, allow_pickle=True, fix_imports=True)
#

###################################
############# Tests ###############
###################################
###################################
