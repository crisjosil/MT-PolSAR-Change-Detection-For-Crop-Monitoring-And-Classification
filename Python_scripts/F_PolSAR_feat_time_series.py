# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 07:54:24 2020
This script creates datacube of PolSAR features for a selected polygon
Select polygon
Crop the multitemporal T to the size of the polygon
Extract the PolSAR features and create a datacube with shape [N,feature,dr,da]
Make zeros the pixels outside the polygon (for the whole datacube of features)
Compute the mean of each PolSAR feature
Plot time series of each feature
@author: Cristian Silva
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt 
import numpy as np
import os
os.chdir('D:\\Juanma - Agrisar 2009\\Full_data\\Python\\Temporal Change detection')
import SAR_utilities_V3a as sar
import rasterio
import fiona
import pandas as pd
from rasterio.mask import mask
import geopandas 
"""
###################################################################################################################################################
Paths
###################################################################################################################################################

"""
tiff_path = "X:\\crs2\\Paper2_Agrisar\\From_June_to_Sept\\"
tiff_name="20090608_Stack_asc_desc.tif" # open geotiff sas raster with rasterio package
shp_path = "X:\\crs2\\Paper2_Agrisar\\From_June_to_Sept\\"
shp_name = "AgriSAR2009.shp"
T_mult_path = "X:\\crs2\\Paper2_Agrisar\\T_Stack\\"
T_mult_name = "T_stack.npy"
"""
###################################################################################################################################################
Loading datacube of array_2D_of_coherency matrices
###########################################################################################################################################################
"""
print("Loading datacube...")
T_mult=np.load(T_mult_path+T_mult_name)
"""
###################################################################################################################################################
read_raster_and_shp
###################################################################################################################################################
"""
raster,Polygons_in_shp = sar.read_raster_and_shp(tiff_path,tiff_name,shp_path,shp_name)
"""
###################################################################################################################################################
Crop T_mult to polygon size
###################################################################################################################################################
"""
#crop_types_IDs=['L-17','P-06','MP-31','B-02','CF-01','CL-03','FL-26','CY-15','O-35','SF-24','FMIX-01','G-05','D-32','W-19']




# list of crops
pol_names=[]
pol_crops=[]
for poly in range(len(Polygons_in_shp)):
    pol_names.append(Polygons_in_shp[poly]['properties']['IDENT']  )  
    pol_crops.append(Polygons_in_shp[poly]['properties']['CROP_TYPE']  )  
#print(pol_names)
# initializing lists 
crop_types = list(set(pol_crops))
"""
###################################################################################################################################################
# loop thorugh polygons and find the biggest 3 of each class
###################################################################################################################################################
"""
df = geopandas.read_file(shp_path + shp_name)
big_IDs=[]
big_IDs_type=[]
for i in range(len(crop_types)):
    temp_df=df[df['CROP_TYPE']==crop_types[i]]
    temp_df.sort_values(by=['AREA_HA'],ascending=False,inplace=True)
    
    big_IDs.append( temp_df.iloc[0]['IDENT'])
    big_IDs_type.append(crop_types[i])
    big_IDs.append( temp_df.iloc[1]['IDENT'])
    big_IDs_type.append(crop_types[i])


df_ts=pd.DataFrame(columns=(['Crop','ID','Feature','stat']+np.arange(1,1+T_mult.shape[0]).tolist()))
df_ts['Crop']=big_IDs_type
df_ts['ID']=big_IDs


feat_list=['VV','VH','HH','VH_VV','HH_VV','HH_VH','L1','alpha1','beta1','L_avg','alpha_avg','entropy','anisotropy'] # repeated
feature_mean=np.zeros([len(big_IDs),T_mult.shape[0],len(feat_list)])
feature_std=np.zeros([len(big_IDs),T_mult.shape[0],len(feat_list)])    
for l in range(len(big_IDs)):   
#l=1
    print("")
    print("###############################################")
    print("Polygon "+str(l)+" of "+str(len(big_IDs))) 
    print("###############################################")
    print("")
    poly=big_IDs[l] # ID of polygon to process
    type_crop=big_IDs_type[l]
    df_ts['Crop'].iloc[0]=type_crop
    df_ts['ID'].iloc[0]=poly
    T_small,numpy_polygon = sar.crop_MT_datacube(poly,raster,Polygons_in_shp,T_mult) # crop the multitemporal datacube to the polygon size
    
    """
    ###################################################################################################################################################
    Extract PolSAR features
    ###################################################################################################################################################
    """
    # PolSAR Features to extract    
    x_max = T_small.shape[1]
    y_max = T_small.shape[2]
    N = T_small.shape[0] # Number of images
    datacube_MT=np.zeros([N,len(feat_list),x_max,y_max],dtype=np.complex64) 
     
    for imagen in range(N):
        print("Img "+str(imagen+1)+" of "+str(N))
        """  
        ###########################################################################################################################################################
        # return the eigenvalue and eigenvector decomposition. 2D array of image size, in which each position contains 3 eigenvelues, or 3x3 eigenvectors #########
        ###########################################################################################################################################################
        """
        L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33=sar.eigendecomposition_vectorized(T_small[imagen,:,:,:,:])    
        """  
        ###########################################################################################################################################################
        # return the alpha, entropy, anisotropy and main scattering mechanishms images (Cloude-Pottier) #########
        ###########################################################################################################################################################
        """
        title1=""
        save=""
        outall=""
        alpha_avg,entropy,anisotropy,R_avg,G_avg,B_avg=sar.Alpha_Entropy_Anisotropy_decomp(L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33,title1,save,outall,factor=3,plot="yes")
           
        #sar.visRGB_from_T(R_avg, G_avg, B_avg,title1,factor=2.5)
        plt.close('all')
        
        # Other features
        print("Plotting alpha/Entropy/Anisotropy results  ...")
        P1=(L1/(L1+L2+L3))
        P2=(L2/(L1+L2+L3))
        P3=(L3/(L1+L2+L3))
        L_avg=(P1*L1)+(P2*L2)+(P3*L3)
        L_avg = np.nan_to_num(L_avg)
        
        #alpha1=np.arccos(U_11)
        #beta1=np.arccos((U_21)/np.sin(alpha1))
        
        # Save the PolSAR features of this image
        datacube_1=np.zeros([len(feat_list),x_max,y_max],dtype=np.complex64)
        datacube_1[0,:,:]= T_small[imagen,:,:,0,0] # VV
        datacube_1[1,:,:]= T_small[imagen,:,:,1,1] # VH
        datacube_1[2,:,:]= T_small[imagen,:,:,2,2] # HH
        datacube_1[3,:,:]= T_small[imagen,:,:,1,1]/T_small[imagen,:,:,0,0] # VH/VV
        datacube_1[4,:,:]= T_small[imagen,:,:,2,2]/T_small[imagen,:,:,0,0] # HH/VV
        datacube_1[5,:,:]= T_small[imagen,:,:,2,2]/T_small[imagen,:,:,1,1] # HH/VH
        datacube_1[6,:,:]= L1
        datacube_1[7,:,:]= np.arccos(U_11) #alpha1
        datacube_1[8,:,:]= np.arccos((U_21)/np.sin(np.arccos(U_11))) #beta1
        datacube_1[9,:,:]= L_avg
        datacube_1[10,:,:]= alpha_avg
        datacube_1[11,:,:]= entropy
        datacube_1[12,:,:]= anisotropy
        # Save the PolSAR features of all images
        datacube_MT[imagen,:,:,:]=datacube_1
        
    # Plot before making zeros outside the polygon
    #sar.visRGB_from_T(datacube_MT[10,1,:,:],  datacube_MT [10,2,:,:], datacube_MT [10,0,:,:],"",factor=2.5)
    #vmin=0
    #vmax=1
    #title=title1+"Entropy"
    #sar.plot_descriptor(np.abs(datacube_MT[10,11,:,:]),vmin,vmax,title)
    
    """
    ###########################################################################################################################################################
    # Make zeros outside the polygon and compute the nan-mean of each PolSAR feature for the selected polygon
    ###########################################################################################################################################################
    """

    for img in range(T_small.shape[0]):
        for polsar_feat in range(len(feat_list)):
            datacube_MT[img,polsar_feat,:,:][numpy_polygon==0]=0
            datacube_MT[img,polsar_feat,:,:]=np.where(datacube_MT[img,polsar_feat,:,:]==0,np.nan,datacube_MT[img,polsar_feat,:,:])
            feature_mean[l,img,polsar_feat]=np.nanmean( datacube_MT[img,polsar_feat,:,:]) # mean
            feature_std [l,img,polsar_feat]= np.nanstd( datacube_MT[img,polsar_feat,:,:]) # std
            
# Plot After making zeros outside the polygon
#sar.visRGB_from_T(datacube_MT[10,1,:,:],  datacube_MT [10,2,:,:], datacube_MT [10,0,:,:],"",factor=2.5)
#vmin=0
#vmax=1
#title=title1+"Entropy"
#sar.plot_descriptor(np.abs(datacube_MT[10,11,:,:]),vmin,vmax,title)
"""
###########################################################################################################################################################
# Plot time series
###########################################################################################################################################################
"""
#for ll in range(len(feat_list)):
#    fig,ax=plt.subplots()
#    ax.scatter(np.arange(0,feature_mean.shape[0]),feature_mean[:,ll],c='k')
#    ax.plot   (np.arange(0,feature_mean.shape[0]),feature_mean[:,ll],'--',c='k',label=feat_list[ll])
#    ax.legend()

lote1=0
lote2=15
for ll in range(len(feat_list)): # plot all features for one polygon
    fig,ax=plt.subplots()
    ax.errorbar(np.arange(0,feature_mean.shape[1]), feature_mean[lote1,:,ll], feature_std[lote1,:,ll], marker='o', mfc='b',mec='b', ms=7, mew=1,label=feat_list[ll]+" "+big_IDs[lote1])   
    ax.errorbar(np.arange(0,feature_mean.shape[1]), feature_mean[lote2,:,ll], feature_std[lote2,:,ll], marker='o', mfc='r',mec='r', ms=7, mew=1,label=feat_list[ll]+" "+big_IDs[lote2])   
    ax.set_xlabel("Image Number")
    ax.legend()

featur=11
fig,ax=plt.subplots()    
for ll in range(len(big_IDs)): # plot one feature for all polygon
    #fig,ax=plt.subplots()
    ax.errorbar(np.arange(0,feature_mean.shape[1]), feature_mean[ll,:,featur], feature_std[0,:,featur], marker='o', mfc='b',mec='b', ms=7, mew=1,label=big_IDs[ll])   
    ax.set_xlabel("Image Number")
    ax.legend()   
    
#for ll in range(len(feat_list)):
#    fig,ax=plt.subplots()
#    ax.errorbar(np.arange(0,feature_mean.shape[0]), feature_mean[:,ll], feature_std[:,ll], marker='o', mfc='b',mec='b', ms=7, mew=1,label=feat_list[ll])   
#    ax.set_xlabel("Image Number")
#    ax.legend() 
#fig,(ax,ax1,ax2)=plt.subplots(3,figsize=(9,9))
#ax.scatter(np.arange(0,backscatter_mean.shape[0]),backscatter_mean[:,0],c='k')
#ax.plot(np.arange(0,backscatter_mean.shape[0]),backscatter_mean[:,0],'--',c='k',label="T00")
#ax.scatter(np.arange(0,backscatter_mean.shape[0]),backscatter_mean[:,1],c='b')
#ax.plot(np.arange(0,backscatter_mean.shape[0]),backscatter_mean[:,1],'--',c='b',label="T11")
#ax.scatter(np.arange(0,backscatter_mean.shape[0]),backscatter_mean[:,2],c='r')
#ax.plot(np.arange(0,backscatter_mean.shape[0]),backscatter_mean[:,2],'--',c='r',label="T22")
#ax1.scatter(np.arange(0,backscatter_mean.shape[0]),(backscatter_mean[:,1]/backscatter_mean[:,0]),c='k')
#ax1.plot(np.arange(0,backscatter_mean.shape[0]),(backscatter_mean[:,1]/backscatter_mean[:,0]),'--',c='k',label="VH/VV")
#ax1.scatter(np.arange(0,backscatter_mean.shape[0]),(backscatter_mean[:,2]/backscatter_mean[:,0]),c='b')
#ax1.plot(np.arange(0,backscatter_mean.shape[0]),(backscatter_mean[:,2]/backscatter_mean[:,0]),'--',c='b',label="HH/VV")
#ax2.scatter(np.arange(0,backscatter_mean.shape[0]),(backscatter_mean[:,2]/backscatter_mean[:,1]),c='k')
#ax2.plot(np.arange(0,backscatter_mean.shape[0]),(backscatter_mean[:,2]/backscatter_mean[:,1]),'--',c='k',label="HH/VH")
#ax.legend()
#ax1.legend()
#ax2.legend()
#plt.tight_layout()