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
################################################# repeat for every beam
#beams=["FQ8W","FQ13W","FQ19W"]  #"AllAll"
#save=1
#for t in range(len(beams)):
#    beam=beams[t]    
################################################# repeat for every polygon in the shp
#for k in range(len(pol_names)):
#k=0
#print("Parcel "+str(k+1)+" of "+str(len(pol_names)))
#poly=k # Type polygon to process
#polygon_A=Polygons_in_shp[poly]["geometry"] # filter the polygon from shp
################################################## Mask a polygon
## use rasterio to mask and obtain the pixels inside the polygon. 
## Because the mask method of rasterio requires an interable object to work, we do:
#polygon_A_list=[polygon_A] 
## out_image in line below is a np array of the polygon desired
#out_image, out_transform = mask(raster, polygon_A_list, crop=True) 
################################################# Get image coord of the corners of the polygon(subset)
#numpy_polygon=out_image[:,:,:][0]
## affine transform of rasterio 
#subset_origin=~raster.transform * (out_transform[2], out_transform[5])
## coordinates of top-left 
#subset_originX = int(subset_origin[0])
#subset_originY = int(subset_origin[1])
## size of the cropped subset after masking
#pixels_in_row=int(numpy_polygon.shape[1])
#pixels_in_col=int(numpy_polygon.shape[0])
#
## Crop the image to some known size otherwise comment (disable) the lines below
## Polygon
#x_min=subset_originY
#x_max=subset_originY+pixels_in_col
#y_min=subset_originX
#y_max=subset_originX+pixels_in_row
 
# Crop the image to some known size otherwise comment (disable) the lines below
# Larger part of the image: Takes about 2-3 hours
y_min=1000
y_max=1500
x_min=1000
x_max=1500

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
import datetime
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
    T11=sar.array2D_of_coherency_matrices_from_stack_SNAP(folder1,basis,in_format,ROI_size,header,datatype,a)
   
    """
    ####################################### T22 ###############################
    """    
    for img_2 in range(N): ####### open image of date j, looping from zero to the nth image     
        a=[]
        for file in os.listdir(folder1):
            if file.endswith(".img"):
                if dates2[img_2] in file:
                    a.append(file) 
        ROI_size=[y_min,y_max,x_min,x_max]
        T22=sar.array2D_of_coherency_matrices_from_stack_SNAP(folder1,basis,in_format,ROI_size,header,datatype,a)
        
        """
        ######################### Bi-Date quad pol Change detection ################
        """
        print("")
        print("Row "+str(img+1)+", Column "+ str(img_2+1))
        print("")
        # Actual change detection
        R_avg[img,img_2,:,:],G_avg[img,img_2,:,:],B_avg[img,img_2,:,:],R_avg[img_2,img,:,:],G_avg[img_2,img,:,:],B_avg[img_2,img,:,:]=sar.bi_date_Quad_pol_CD(T11,T22)


"""
Plot some change matrices
"""

def plot_CM(pix_i,pix_j,N):
    CM=np.zeros([N,N,3]) # change matrix    
    for img in range(N):
        for img_2 in range(N):
            CM[img,img_2,0]=np.nanmean(R_avg[img,img_2,pix_i,pix_j]) # mean
            CM[img,img_2,1]=np.nanmean(G_avg[img,img_2,pix_i,pix_j]) # mean
            CM[img,img_2,2]=np.nanmean(B_avg[img,img_2,pix_i,pix_j]) # mean    
    
    fig,ax=plt.subplots()
    ax.imshow(CM*4)

pix_i=578
pix_j=1119
plot_CM(pix_i,pix_j,N)

pix_i=586
pix_j=1007
plot_CM(pix_i,pix_j,N)

pix_i=774
pix_j=1082
plot_CM(pix_i,pix_j,N)

pix_i=880
pix_j=1110
plot_CM(pix_i,pix_j,N)

pix_i=1045
pix_j=1012
plot_CM(pix_i,pix_j,N)

        
"""
plot changes from first image to the rest

Because We havent removed areas not coincident, they may appear in the increase/decrease plots
"""
factor=7
size = np.shape(R_avg[0,0,:,:])           
for img in range(len(dates2)-1):
    print(str(img+1))
    img_a=1
    img_b=img+1    
    
    iRGB = np.zeros([size[0],size[1],3])
    iRGB[:,:,0] = np.abs(R_avg[img_a,img_b,:,:])/((np.nanmean(np.abs(R_avg[img_a,img_b,:,:])))*factor)
    iRGB[:,:,1] = np.abs(G_avg[img_a,img_b,:,:])/((np.nanmean(np.abs(G_avg[img_a,img_b,:,:])))*factor)
    iRGB[:,:,2] = np.abs(B_avg[img_a,img_b,:,:])/((np.nanmean(np.abs(B_avg[img_a,img_b,:,:])))*factor)
    iRGB=np.nan_to_num(iRGB)   
    
    iRGB_dec = np.zeros([size[0],size[1],3])
    iRGB_dec[:,:,0] = np.abs(R_avg[img_b,img_a,:,:])/((np.nanmean(np.abs(R_avg[img_b,img_a,:,:])))*factor)
    iRGB_dec[:,:,1] = np.abs(G_avg[img_b,img_a,:,:])/((np.nanmean(np.abs(G_avg[img_b,img_a,:,:])))*factor)
    iRGB_dec[:,:,2] = np.abs(B_avg[img_b,img_a,:,:])/((np.nanmean(np.abs(B_avg[img_b,img_a,:,:])))*factor)
    iRGB_dec=np.nan_to_num(iRGB_dec)  
    
    title="Scattering mechanisms that increased strength (dates: 1 and "+str(img+2)
    title_dec="Scattering mechanisms that decreased strength (dates: 1 and "+str(img+2)
         
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,figsize=(19, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(iRGB)*3) 
    ax2.set_title(title_dec)
    ax2.imshow(np.abs(iRGB_dec)*3) 
    ax1.axis("off")
    ax2.axis("off")
    plt.tight_layout()
##     

# Interpolate the change matrix
#CM1=np.nan_to_num(CM) # To interpolate we need to replace the nan of diagonals to zeros
#days=100 # days to interpolate
#CM_interpol=np.zeros([days,days,3])
#CM_interpol[:,:,0]=sar.interpolate_matrix(CM1[:,:,0], days)
#CM_interpol[:,:,1]=sar.interpolate_matrix(CM1[:,:,1], days)
#CM_interpol[:,:,2]=sar.interpolate_matrix(CM1[:,:,2], days)
#
#outall=""
#title="CM_"+pol_names[k]
#sar.Plot_all_change_matrices(CM1,title,save,outall,factor)
#
#outall=""
#title="CM_Interpol_"+pol_names[k]
#sar.Plot_all_change_matrices(CM_interpol,title,save,outall,factor)

#plt.close()

CM_Datacube=np.zeros((x_max,y_max,N,N,3))
for i in range(len(dates2)):
    for j in range(len(dates2)):
        CM_Datacube[:,:,i,j,0]=R_avg[i,j,:,:]
        CM_Datacube[:,:,i,j,1]=G_avg[i,j,:,:]
        CM_Datacube[:,:,i,j,2]=B_avg[i,j,:,:]
        
        

##############################################
# load ground truth as in the stack
#layer_name = "Labels_asc.npy" #"Labels_asc.npy"
#print("loading "+layer_name)
#labels=np.load('D:\\Paper2\\Ascending_only_V1\\np_arrays\\'+layer_name,allow_pickle=True) 
## load Train_test_flags
#layer_name = "Train_test_flags_asc.npy"#"Areas_asc_FQ2.npy"
#print("loading "+layer_name)
#tr_ts_arr=np.load('D:\\Paper2\\Ascending_only_V1\\np_arrays\\'+layer_name,  allow_pickle=True)        
################################## Count number of training and testing pixels
#n_samples_train=0
#n_samples_test=0
#training_samples=0
#testing_samples=0
#for i in range(tr_ts_arr.shape[0]):
#    for j in range(tr_ts_arr.shape[1]):
#        if tr_ts_arr[i,j]==0:
#            training_samples=training_samples+1
#        elif tr_ts_arr[i,j]==1:
#            testing_samples=testing_samples+1        
#"""
###################################
######### Create train and test sets for ML classification
####################################
#"""
#n_feat = CM_Datacube[0,0,:,:,:].flatten().shape[0]
## Empty arrays for ML classification    
#X_train = np.zeros([training_samples,n_feat])
#y_train_0 = np.zeros(training_samples)
#X_test = np.zeros([testing_samples,n_feat])
#y_test = np.zeros(testing_samples)
#
## aux variables
#nan_pos_list_train=[]
#nan_pos_list_test=[]
#n_samples_train=-1
#n_samples_test=-1
## Determine if a pixel is training or test and add its corresponding features and crop type label to the empty arrays
#for h in range(tr_ts_arr.shape[0]):
#    print(str(int(100*h/tr_ts_arr.shape[0]))+str('%')) # % of progress
#    for r in range(tr_ts_arr.shape[1]):
#        if tr_ts_arr[h,r]==0: # is it a training pixel?
#            n_samples_train=n_samples_train+1  # used as index in X_train and y_train_0 array
#            var=CM_Datacube[h,r,:,:,:].flatten() # flattened change matrix of the pixel h,r
#            X_train[n_samples_train,:] = var
#            y_train_0[n_samples_train]   = labels[h,r] # get the corresponding class label
#            if np.isnan(np.sum(var))==True:  # if there is a nan in the feature vector, save the position to remove it later
#                nan_pos_list_train.append(n_samples_train)
#        
#        elif tr_ts_arr[h,r]==1: # is it a testing pixel?
#            n_samples_test=n_samples_test+1
#            var=CM_Datacube[h,r,:,:,:].flatten() # flattened change matrix of the pixel h,r
#            X_test[n_samples_test,:] = var
#            y_test[n_samples_test] = labels[h,r]
#            if np.isnan(np.sum(var))==True:
#                nan_pos_list_test.append(n_samples_test)
# # Delete the rows with nans                
#X_train2=np.delete(X_train, nan_pos_list_train,axis=0)
#y_train2=np.delete(y_train_0, nan_pos_list_train,axis=0)
#X_test2=np.delete(X_test, nan_pos_list_test,axis=0)
#y_test2=np.delete(y_test, nan_pos_list_test,axis=0)
#
#
##### Assign grass and mixed hay classes as more general mixed pasture class
#y_train=y_train2.copy() 
#y_train=np.where(y_train2==8, 11,y_train)
#y_train=np.where(y_train2==10,11,y_train)
#print(Counter(y_train))
#
#y_test=y_test2.copy() 
#y_test=np.where(y_test2==8, 11,y_test)
#y_test=np.where(y_test2==10,11,y_test)
#print(Counter(y_test))

location="D:\Paper2\Ascending_only_V1\\np_arrays\\Change_matrix\\"
layer_name = "CM_Datacube" #
np.save(location+layer_name, CM_Datacube, allow_pickle=True, fix_imports=True)
#layer_name = "X_train" #
#np.save(location+layer_name, X_train2, allow_pickle=True, fix_imports=True)
#layer_name = "y_train"
#np.save(location+layer_name, y_train, allow_pickle=True, fix_imports=True)
#layer_name = "X_test"
#np.save(location+layer_name, X_test2, allow_pickle=True, fix_imports=True)
#layer_name = "y_test"
#np.save(location+layer_name, y_test, allow_pickle=True, fix_imports=True)

"""
##############################################################
################## Fill the diagonals ########################
#################################################################
"""
#layer_name = "datacube_MT_asc.npy"#"Areas_asc_FQ2.npy"
#print("loading "+layer_name)
#datacube_MT=np.load('D:\\Paper2\\Ascending_only_V1\\np_arrays\\'+layer_name,  allow_pickle=True)
#
#layer_name = "CM_Datacube.npy"#"Areas_asc_FQ2.npy"
#print("loading "+layer_name)
#CM_Datacube=np.load("D:\Paper2\Ascending_only_V1\\np_arrays\\Change_matrix\\"+layer_name,  allow_pickle=True) 
#
#print(datacube_MT.shape)
#print(CM_Datacube.shape)
#N = 7 # number of images in the stack
#for i in range(N):
#    CM_Datacube[:,:,i,i,0]=datacube_MT[i,13,:,:]
#    CM_Datacube[:,:,i,i,1]=datacube_MT[i,14,:,:]
#    CM_Datacube[:,:,i,i,2]=datacube_MT[i,15,:,:]
#    
#pix_i=880
#pix_j=1110
#plot_CM(pix_i,pix_j,N)
#
#pix_i=1045
#pix_j=1012
#plot_CM(pix_i,pix_j,N)
#
#
#
#location="D:\Paper2\Ascending_only_V1\\np_arrays\\Change_matrix\\"
#layer_name = "CM_Datacube_with_diag" #
#np.save(location+layer_name, CM_Datacube, allow_pickle=True, fix_imports=True)
#from sklearn.preprocessing import MinMaxScaler # [0,1]   # recommended for NNs
#from sklearn.decomposition import PCA
#from mpl_toolkits.mplot3d import Axes3D
#from imblearn.under_sampling import (RandomUnderSampler)
#print("Scaling ...")         
#sc = MinMaxScaler()
#X_train = sc.fit_transform(X_train2)
##X_val1 = sc.transform(X_val)
#X_test = sc.transform(X_test2)
#print("done")
#
#def plot_class_separability(samples,sampling_strategy,X_train, y_train,cols,labels):
#    under_s = RandomUnderSampler (sampling_strategy = sampling_strategy, random_state=42)
#    XX, yy = under_s.fit_resample(X_train, y_train)
#    print((Counter(yy)))
#    pca = PCA(3)  # project from 64 to 2 dimensions
#    projected = pca.fit_transform(XX)
#    
#    ini=-samples
#    end=-samples
#    fig = plt.figure(figsize=(12,8))
#    ax = fig.add_subplot(111, projection='3d')
#    for i in range(np.unique(yy).shape[0]):
#        ini=ini+samples
#        end = end + samples
#        xs=projected[:, 0][ini:ini+samples], 
#        ys=projected[:, 1][ini:ini+samples]
#        zs=projected[:, 2][ini:ini+samples]
#        ax.scatter(xs, ys, zs, c=cols[i],alpha=1,s=20, label = labels[i])
#        #fig.colorbar(p).set_label('crop type')
#        ax.set_xlim(-1.5,1.5)
#        ax.set_ylim(-1.5,1.5)
#        ax.set_zlim(-1.5,1.5)
#        ax.legend()
#
#Reduced_Labels = ['Barley','Canaryseed','Canola', 'Durum Wheat', 'Field Pea', 'Flax',
#                  'Lentil','Mixed Pasture', 'Oat', 'Spring Wheat']
#
## All classes
#Reduced_Labels=Reduced_Labels.copy()
#Reduced_Labels = [w.replace('Durum Wheat', 'Durum_Wheat') for w in Reduced_Labels]
#Reduced_Labels = [w.replace('Field Pea', 'Field_Pea') for w in Reduced_Labels]
#Reduced_Labels = [w.replace('Mixed Pasture', 'Mixed_Pasture') for w in Reduced_Labels]
#Reduced_Labels = [w.replace('Spring Wheat', 'Spring_Wheat') for w in Reduced_Labels]
#print(Reduced_Labels)
#        
#samples=200
#sampling_strategy={1: samples, 2: samples, 3: samples,5: samples, 6: samples, 7: samples,
#                   9: samples, 11: samples, 12: samples, 14: samples}
#cols = ['red', 'green','purple', 'blue', 'orange', 'k',
#                          'lime', 'yellow','cyan', 'fuchsia']
#plot_class_separability(samples,sampling_strategy,X_train, y_train,cols,Reduced_Labels)
#
#""" 
#Reduced classes
#"""
#Reduced_Labels_red=["Cereals","Canola","field Pea","Flax","Lentil","Pasture"]
#
#df_metrics_red=pd.DataFrame(index=['f1_micro','f1_macro','f1_weighted','G-mean_micro','G-mean_macro','G-mean_weigthed','Balanced accuracy'])
#
#y_train_red=y_train.copy() 
#y_train_red=np.where(y_train==1,0,y_train_red) # cereals: Barley
#y_train_red=np.where(y_train==2,0,y_train_red) # cereals: Canaryseed
#y_train_red=np.where(y_train==5,0,y_train_red) # cereals: Durum wheat
#y_train_red=np.where(y_train==12,0,y_train_red) # cereals: oat
#y_train_red=np.where(y_train==14,0,y_train_red) # cereals: Spring wheat
#y_train_red=np.where(y_train==3,1,y_train_red) # Canola
#y_train_red=np.where(y_train==6,2,y_train_red) # Pea
#y_train_red=np.where(y_train==7,3,y_train_red) # Flax
#y_train_red=np.where(y_train==9,4,y_train_red) # Lentil
#y_train_red=np.where(y_train==11,5,y_train_red) # Mixed Pasture
#
#y_test_red=y_test.copy() 
#y_test_red=np.where(y_test==1,0,y_test_red) # cereals: Barley
#y_test_red=np.where(y_test==2,0,y_test_red) # cereals: Canaryseed
#y_test_red=np.where(y_test==5,0,y_test_red) # cereals: Durum wheat
#y_test_red=np.where(y_test==12,0,y_test_red) # cereals: oat
#y_test_red=np.where(y_test==14,0,y_test_red) # cereals: Spring wheat
#y_test_red=np.where(y_test==3,1,y_test_red) # Canola
#y_test_red=np.where(y_test==6,2,y_test_red) # Pea
#y_test_red=np.where(y_test==7,3,y_test_red) # Flax
#y_test_red=np.where(y_test==9,4,y_test_red) # Lentil
#y_test_red=np.where(y_test==11,5,y_test_red) # Mixed Pasture
#
#             
#cases_red =['RF_red','BRF_red','Under_BRF_red','Imb_Under_BRF_red']
#
## reduced classes
#samples=200
#sampling_strategy={0: samples, 1: samples, 2: samples,3: samples, 4: samples, 5: samples}
#cols = ['red', 'k','purple', 'blue', 'orange', 'green']
#plot_class_separability(samples,sampling_strategy,X_train, y_train_red,cols,Reduced_Labels_red)


