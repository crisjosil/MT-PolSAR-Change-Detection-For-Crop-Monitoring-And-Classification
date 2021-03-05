# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:00:05 2021

@author: crs2
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt 
import numpy as np
import os
import SAR_utilities_V3a as sar
import pandas as pd
import fiona
import rasterio
from rasterio.mask import mask
#import scipy 

#%%  
# plot CM funcitons
def plot_CM_interpolated(dates2,CM_interpol,factor=0.3,save=0,title="",outall=""):
    tt= np.arange(len(dates2))*10 + 0
    fig,ax=plt.subplots(1,figsize=(11,11))
    ax.set_xticks(tt+5)
    ax.set_yticks(tt+5)
    ax.set_xticklabels(dates2,fontsize=20)
    ax.set_yticklabels(dates2,fontsize=20)
    for i in range(len(dates2)):
        ax.axvline(tt[i], ls='--', color='k',linewidth=0.5)
        ax.axhline(tt[i], ls='--', color='k',linewidth=0.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    ax.imshow(CM_interpol*factor)
    plt.tight_layout()
    if save ==1:
        print("saving "+title+" ..." )
        outall1= outall+title+'.png'
        fig.savefig(outall1, bbox_inches='tight')

def plot_CM(dates2,CM,factor=0.3,save=0,title="",outall=""):
    fig,ax=plt.subplots(1,figsize=(11,11))
    tt= np.arange(len(dates2))
    ax.set_xticks(tt+0)
    ax.set_yticks(tt+0)
    ax.set_xticklabels(dates2,fontsize=18)
    ax.set_yticklabels(dates2,fontsize=18)
    #for i in range(len(files)):
    #    ax.axvline(tt[i], ls='--', color='k',linewidth=0.5)
    #    ax.axhline(tt[i], ls='--', color='k',linewidth=0.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    ax.imshow(CM*factor)
    plt.tight_layout()
    if save ==1:
        print("saving "+title+" ..." )
        outall1= outall+title+'.png'
        fig.savefig(outall1, bbox_inches='tight')
#%%
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
raster = rasterio.open(tiff_path+name)
################################################# read shp
# Read shapefile with fiona package
Polygons_in_shp=fiona.open("D:\\Paper2\\Ascending_only_V1\\AgriSAR2009.shp", "r")
#read the polygon names, crop types and areas and save them in a list
pol_names=[]
pol_crops=[]
pol_areas=[]
for poly in range(len(Polygons_in_shp)):
    pol_names.append(Polygons_in_shp[poly]['properties']['IDENT']  )  
    pol_crops.append(Polygons_in_shp[poly]['properties']['CROP_TYPE']  )  
    pol_areas.append(Polygons_in_shp[poly]['properties']['AREA_HA']  )    
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
datatype = 'float32'
basis="P"     # Lexicographic (L) or Pauli (P)
in_format="img" # Bin or img

# instantiate class to perform eigendecompositions indicating that we will use a covariance matrix correspondint to difference of two images
application = "Difference change detection"
#eigen = sar.eigendecompositions(application)

#create dataframe with same info    
df=pd.DataFrame(index=np.arange(1,len(pol_names)+1),columns=['IDENT','CROP_TYPE','AREA_HA'])
df['IDENT']=pol_names
df['CROP_TYPE']=pol_crops
df['AREA_HA']=pol_areas

save=0
crop_types = list(set(pol_crops))
#crop_types_IDs=['L-17','P-06','MP-31','B-02','CF-01','CL-03','FL-26','CY-15','O-35','SF-24','FMIX-01','G-05','D-32','W-19'] 

#%%
#crop_types_IDs=['B-02','CL-23','P-10','FL-26','L-17','MP-23','G-03','CY-25'] #262 Ha
crop_types_IDs=['B-02','CL-23','P-10']
#crop_types_IDs=['CL-23']
#for each crop type, select the parcel with largest area 
application = "Difference change detection"
eigen = sar.eigendecompositions(application)
for ctype in range(len(crop_types_IDs)):  
    print(crop_types_IDs[ctype])
    application = "Difference change detection"
    eigen = sar.eigendecompositions(application)
#print("Crop type "+str(ctype+1)+" of "+str(len(crop_types)+1))
#temp_crop=df[df['CROP_TYPE']==crop_types[ctype]]
    
    temp_crop=df[df['IDENT']==crop_types_IDs[ctype]]
    temp_crop_sort=temp_crop.sort_values(by=['AREA_HA'], ascending=False)
    poly=temp_crop_sort.index[0]-1 # Type polygon to process
    #Alfalfa=157-1
    #Canaryseed=62-1
    #Canola=
    #poly=62-1
    ################################################# Crop the polygon
    polygon_A=Polygons_in_shp[int(poly)]["geometry"] # filter the polygon from shp
    ################################################# Mask the polygon
    # use rasterio to mask and obtain the pixels inside the polygon. 
    # Because the mask method of rasterio requires an interable object to work, we do:
    polygon_A_list=[polygon_A] 
    # out_image in line below is a np array of the polygon desired
    out_image, out_transform = mask(raster, polygon_A_list, crop=True) 
    ################################################ Get image coord of the corners of the polygon(subset)
    numpy_polygon=out_image[:,:,:][0]
    # affine transform of rasterio 
    subset_origin=~raster.transform * (out_transform[2], out_transform[5])
    # coordinates of top-left 
    subset_originX = int(subset_origin[0])
    subset_originY = int(subset_origin[1])
    # size of the cropped subset after masking
    pixels_in_row=int(numpy_polygon.shape[1])
    pixels_in_col=int(numpy_polygon.shape[0])
    
    # Crop the image to some known size otherwise comment (disable) the lines below
    # Polygon
    x_min=subset_originY
    x_max=subset_originY+pixels_in_col
    y_min=subset_originX
    y_max=subset_originX+pixels_in_row
    # Compute image size
    ROI_size=[y_min,y_max,x_min,x_max]
    da=(x_max-x_min) # cols
    dr=(y_max-y_min) # rows
    # Empty array to save outputs of multitemporal change detection
    R_avg=np.zeros([N,N,da,dr])
    G_avg=np.zeros([N,N,da,dr])
    B_avg=np.zeros([N,N,da,dr])
    CM=np.zeros([N,N,3]) # change matrix
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
        
        T11=sar.array2D_of_coherency_matrices_from_stack_SNAP(folder1,basis,in_format,
                                                              ROI_size,header,datatype,a)
        """
        ####################################### T22 ###############################
        """    
        for img_2 in range(N): ####### open image of date j, looping from zero to the nth image     
            a=[]
            for file in os.listdir(folder1):
                if file.endswith(".img"):
                    if dates2[img_2] in file:
                        a.append(file) 
            T22=sar.array2D_of_coherency_matrices_from_stack_SNAP(folder1,basis,in_format,
                                                                  ROI_size,header,datatype,a)
            """
            ######################### Bi-Date quad pol Change detection ################
            """
            print("")
            print("Row "+str(img+1)+", Column "+ str(img_2+1))
            print("Processing added scattering mechanisms...")
            Tc = (T22 - T11)      
            List_RGB = eigen.gral_eigendecomposition(Tc) # to store eigendecomposition results in the class
            R_avg[img,img_2,:,:],G_avg[img,img_2,:,:],B_avg[img,img_2,:,:] = eigen.vis(eigen.L1_inc,eigen.L2_inc,eigen.L3_inc,add_or_remove='added') # added SMs
            R_avg[img_2,img,:,:],G_avg[img_2,img,:,:],B_avg[img_2,img,:,:] = eigen.vis(eigen.L1_dec,eigen.L2_dec,eigen.L3_dec,add_or_remove='removed') # removed SMs
             
    # Make zero the pixels outside the polygon boundaries,Replace zeros with nan,compute nanmean   
    for img in range(N):
        for img_2 in range(N):
                R_avg[img,img_2,:,:][numpy_polygon==0]=0
                G_avg[img,img_2,:,:][numpy_polygon==0]=0
                B_avg[img,img_2,:,:][numpy_polygon==0]=0
                # Replace zeros with nan
                R_avg[img,img_2,:,:]=np.where(R_avg[img,img_2,:,:]==0,np.nan,R_avg[img,img_2,:,:])
                G_avg[img,img_2,:,:]=np.where(G_avg[img,img_2,:,:]==0,np.nan,G_avg[img,img_2,:,:])
                B_avg[img,img_2,:,:]=np.where(B_avg[img,img_2,:,:]==0,np.nan,B_avg[img,img_2,:,:])
                # compute nanmean 
                CM[img,img_2,0]=np.nanmean(R_avg[img,img_2,:,:]) # mean
                CM[img,img_2,1]=np.nanmean(G_avg[img,img_2,:,:]) # mean
                CM[img,img_2,2]=np.nanmean(B_avg[img,img_2,:,:]) # mean
    
   # Interpolate the change matrix
    CM1=np.nan_to_num(CM) # To interpolate we need to replace the nan of diagonals to zeros
    CM1[:,:,0] = CM1[:,:,0]/((np.abs(np.nanmean(CM1[:,:,0]))))
    CM1[:,:,1] = CM1[:,:,1]/((np.abs(np.nanmean(CM1[:,:,1]))))
    CM1[:,:,2] = CM1[:,:,2]/((np.abs(np.nanmean(CM1[:,:,2]))))
    outall="D:\\Juanma - Agrisar 2009\\My_results\\Change_matrices\\Original\\"
    title="CM_"+crop_types_IDs[ctype]
    plot_CM(dates2,CM1,factor=0.3,save=1,title=title,outall=outall)
    
    
    # this
    change_matrix = np.nan_to_num(CM)
    days=len(dates2) *10 # days to interpolate
    CM_interpol=np.zeros([days,days,3])
    CM_interpol[:,:,0]=sar.interpolate_matrix(change_matrix[:,:,0], days)
    CM_interpol[:,:,1]=sar.interpolate_matrix(change_matrix[:,:,1], days)
    CM_interpol[:,:,2]=sar.interpolate_matrix(change_matrix[:,:,2], days)
    
    CM_interpol[:,:,0]=    CM_interpol[:,:,0]/((np.abs(np.nanmax(CM_interpol[:,:,0]))))
    CM_interpol[:,:,1]=    CM_interpol[:,:,1]/((np.abs(np.nanmax(CM_interpol[:,:,1]))))
    CM_interpol[:,:,2]=    CM_interpol[:,:,2]/((np.abs(np.nanmax(CM_interpol[:,:,2])))) 
    outall="D:\\Juanma - Agrisar 2009\\My_results\\Change_matrices\\Interpolated\\"
    title="CM_"+crop_types_IDs[ctype]
    plot_CM_interpolated(dates2,CM_interpol,factor=1,save=1,title=title,outall=outall) 
#    
#    """
#    #################################################
#    ################# Diagonals
#    #################################################
#    """       
#    application = "Single image"
#    eigen = sar.eigendecompositions(application)
#            
#    for img in range(N): ####### open image of date i and fix it   
#        print(str(img+1))
#        """
#        ####################################### T11 ###############################
#        """
#        a=[]
#        for file in os.listdir(folder1):
#            if file.endswith(".img"):
#                if dates2[img] in file:
#                    a.append(file)    
#        ROI_size=[y_min,y_max,x_min,x_max]
#        T=sar.array2D_of_coherency_matrices_from_stack_SNAP(folder1,basis,in_format,ROI_size,header,datatype,a)
#        R, G, B = eigen.gral_eigendecomposition(T)
#        R_avg[img,img,:,:] = R
#        G_avg[img,img,:,:] = G
#        B_avg[img,img,:,:] = B
#        
#    # Make zero the pixels outside the polygon boundaries,Replace zeros with nan,compute nanmean   
#    for img in range(N):
#        for img_2 in range(N):
#                R_avg[img,img_2,:,:][numpy_polygon==0]=0
#                G_avg[img,img_2,:,:][numpy_polygon==0]=0
#                B_avg[img,img_2,:,:][numpy_polygon==0]=0
#                # Replace zeros with nan
#                R_avg[img,img_2,:,:]=np.where(R_avg[img,img_2,:,:]==0,np.nan,R_avg[img,img_2,:,:])
#                G_avg[img,img_2,:,:]=np.where(G_avg[img,img_2,:,:]==0,np.nan,G_avg[img,img_2,:,:])
#                B_avg[img,img_2,:,:]=np.where(B_avg[img,img_2,:,:]==0,np.nan,B_avg[img,img_2,:,:])
#                # compute nanmean 
#                CM[img,img_2,0]=np.nanmean(R_avg[img,img_2,:,:]) # mean
#                CM[img,img_2,1]=np.nanmean(G_avg[img,img_2,:,:]) # mean
#                CM[img,img_2,2]=np.nanmean(B_avg[img,img_2,:,:]) # mean
    # repeat plot lines if needed
#%%
plot_CM_interpolated(dates2,CM_interpol,factor=1,save=0,title=title,outall=outall) 
