# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:18:04 2018

@author: Cristian Silva

This script:
    Opens a shapefile and reads the geocoordinates of the polygons in it
    For a selected polygon(s) transforms geocoordinates to image coordinates, 
    Opens the RADARSAT-2 images, performs pixelwise multitemporal change detection for the polygon
    Replaces the pixels outside the polygon with nan and computes the nanmean per channel (scattering mechanism)
    Creates the change matrix for this polygon(s) 
    Interpolates the change matrix to a given number of days
    Saves results if save = 1
"""

import numpy as np
import matplotlib.pyplot as plt 
#import sys
#sys.path.insert(1, 'D:\\Juanma - Agrisar 2009\\Full_data\\Python\\Temporal Change detection\\')
import SAR_utilities_V3a as sar
import fiona
#import scipy 
import rasterio
from rasterio.mask import mask

# plot CM funcitons
def plot_CM_interpolated(dates2,CM_interpol,factor=0.3,save=0,title="",outall=""):
    tt= np.arange(len(dates2))*10 + 0
    fig,ax=plt.subplots(1,figsize=(12,9))
    ax.set_xticks(tt+5)
    ax.set_yticks(tt+5)
    ax.set_xticklabels(dates2,fontsize=14)
    ax.set_yticklabels(dates2,fontsize=14)
    for i in range(len(files)):
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
    fig,ax=plt.subplots(1,figsize=(12,9))
    tt= np.arange(len(dates2))
    ax.set_xticks(tt+0)
    ax.set_yticks(tt+0)
    ax.set_xticklabels(dates2,fontsize=14)
    ax.set_yticklabels(dates2,fontsize=14)
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
        
#folder="D:\\Datasets\\Agrisar_2009\\Mosaic\\C3\\"
#y_max=1900
#x_max=3380
################################################# Read tif raster of the SAR image ##############################
# Saved after doing the preprocessing. This contains geo info to map from 
# geocoordinates to image coordinates
tiff_path = "D:\\Juanma\\Seville 2014\\FQ13W\\2014-06-22.rds2\\"
name="final_test_no_rev.tif"
# open geotiff sas raster with rasterio package
raster = rasterio.open(tiff_path+name)
################################################# read shp
# Read shapefile with fiona package
Polygons_in_shp=fiona.open('D:\\Juanma\\Parcels_A_to_F_UTM29N.shp', "r")
#read the polygon names and save them in a list
pol_names=[]
for poly in range(len(Polygons_in_shp)):
    pol_names.append(Polygons_in_shp[poly]['properties']['Name']  )  
print(pol_names)
################################################# repeat for every beam
#beams=["FQ8W","FQ13W","FQ19W"]  #"AllAll"
#beams=["FQ8W"]  #"AllAll"
#dates2 = ['22May2014','15June2014','09July2014','02August2014','26August2014','19Sept2014']
#beams=["FQ13W"]
#dates2 = ['22June2014','16July2014','09August2014','02Sept2014','26Sept2014']
beams=["FQ19W"]
dates2 = ['05June2014','29June2014','23July2014','16August2014','09Sept2014']
save=1
application = "Difference change detection"
eigen = sar.eigendecompositions(application)
for t in range(len(beams)):
    beam=beams[t]    
################################################# repeat for every polygon in the shp
    #for k in range(len(pol_names)):
    for k in range(1):
        #k=0
        print("Parcel "+str(k+1)+" of "+str(len(pol_names)))
        poly=k # Type polygon to process
        polygon_A=Polygons_in_shp[poly]["geometry"] # filter the polygon from shp
        ################################################# Mask a polygon
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
         
        # Crop the image to some known size otherwise comment (disable) the lines below
        # Larger part of the image: Takes about 2-3 hours
        #y_min=3000
        #y_max=6200
        #x_min=3500
        #x_max=6500
        
        # Compute image size
        ROI_size=[y_min,y_max,x_min,x_max]
        da=(x_max-x_min) # cols
        dr=(y_max-y_min) # rows
        
        files=sar.select_images_office(beam)[0]
        #test
        #folder=files[0]
        #cols,rows,header=sar.read_config_file(folder)
        #datatype = 'float32'
        #basis="L"     # Lexicographic (L) or Pauli (P)
        #in_format="Bin" # .Bin
        #T11=sar.array2D_of_coherency_matrices(folder,basis,in_format,ROI_size,header,datatype)
        #sar.visRGB(T11[:,:,1,1], T11[:,:,2,2], T11[:,:,0,0],"")
        
        # Empty array to save outputs of multitemporal change detection
        R_avg=np.zeros([len(files),len(files),da,dr])
        G_avg=np.zeros([len(files),len(files),da,dr])
        B_avg=np.zeros([len(files),len(files),da,dr])
        CM=np.zeros([len(files),len(files),3]) # change matrix
        
        # Multitemporal change detection of the cropped region  
        # First row of change matrix, Open the image 1 (i) and do change detection with respecto to all other images in the stack
        # Second row: Open next image (i+1) in the stack and do change detection 
        # Continue for the n images in the stack
        for img in range(len(files)): ####### open image of date i and fix it   
            print(str(img+1))
            """
            ####################################### T11 ###############################
            """
            #img=0
            folder=files[img] # path of the image
            cols,rows,header=sar.read_config_file(folder)
            datatype = 'float32'
            basis="L"     # Lexicographic (L) or Pauli (P)
            in_format="Bin" # .Bin
            T11=sar.array2D_of_coherency_matrices(folder,basis,in_format,ROI_size,header,datatype)
            
            """
            ####################################### T22 ###############################
            """    
            for img_2 in range(len(files)): ####### open image of date j, looping from zero to the nth image     
                #folder="D:\\Datasets\\C3\\"
                folder=files[img_2]
                x_max,y_max,header=sar.read_config_file(folder)
                T22=sar.array2D_of_coherency_matrices(folder,basis,in_format,ROI_size,header,datatype)
                
                """
                ######################### Bi-Date quad pol Change detection ################
                """
                print("")
                print("Row "+str(img+1)+", Column "+ str(img_2+1))
                print("")
                # Actual change detection
                #R_avg[img,img_2,:,:],G_avg[img,img_2,:,:],B_avg[img,img_2,:,:],R_avg[img_2,img,:,:],G_avg[img_2,img,:,:],B_avg[img_2,img,:,:]=sar.bi_date_Quad_pol_CD(T11,T22)
                Tc = (T22 - T11)      
                List_RGB = eigen.gral_eigendecomposition(Tc) # to store eigendecomposition results in the class
                R_avg[img,img_2,:,:],G_avg[img,img_2,:,:],B_avg[img,img_2,:,:] = eigen.vis(eigen.L1_inc,eigen.L2_inc,eigen.L3_inc,add_or_remove='added') # added SMs
                R_avg[img_2,img,:,:],G_avg[img_2,img,:,:],B_avg[img_2,img,:,:] = eigen.vis(eigen.L1_dec,eigen.L2_dec,eigen.L3_dec,add_or_remove='removed') # removed SMs
                
        # Make zero the pixels outside the polygon boundaries,Replace zeros with nan,compute nanmean   
        for img in range(len(files)):
            for img_2 in range(len(files)):
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
        outall="D:\\Juanma\\Seville 2014\\"+beam+"\\Change_matrices\\Original\\"
        title="CM_"+pol_names[k]
        plot_CM(dates2,CM1,factor=0.3,save=1,title=title,outall=outall)
        
        CM1=np.nan_to_num(CM)
        days=len(files) *10 # days to interpolate
        CM_interpol=np.zeros([days,days,3])
        CM_interpol[:,:,0]=sar.interpolate_matrix(CM1[:,:,0], days)
        CM_interpol[:,:,1]=sar.interpolate_matrix(CM1[:,:,1], days)
        CM_interpol[:,:,2]=sar.interpolate_matrix(CM1[:,:,2], days)
        
        CM_interpol[:,:,0]=    CM_interpol[:,:,0]/((np.abs(np.nanmean(CM_interpol[:,:,0]))))
        CM_interpol[:,:,1]=    CM_interpol[:,:,1]/((np.abs(np.nanmean(CM_interpol[:,:,1]))))
        CM_interpol[:,:,2]=    CM_interpol[:,:,2]/((np.abs(np.nanmean(CM_interpol[:,:,2]))))  
        outall="D:\\Juanma\\Seville 2014\\"+beam+"\\Change_matrices\\Interpol\\"
        title="CM_Interpol_"+pol_names[k]
        plot_CM_interpolated(dates2,CM_interpol,factor=0.3,save=1,title=title,outall=outall)        
        #plt.close()
        

