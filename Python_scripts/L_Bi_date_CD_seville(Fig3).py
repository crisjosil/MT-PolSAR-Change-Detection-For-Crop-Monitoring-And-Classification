# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:16:08 2021
Compute the pairwise change detection using the difference of covariance matrices
@author: Cristian Silva
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt 
import numpy as np
#import geopandas as gpd
#import os
import SAR_utilities_V3a as sar



"""
######################### Beam ################
"""
 ##############################################'FQ13W'  
#beam = 'FQ8W'     
#dates=["2014-05-22","2014-06-15","2014-07-09","2014-08-02","2014-08-26","2014-09-19"]
 ##############################################'FQ13W'  
#beam = 'FQ13W'     
#dates=["2014-06-22","2014-07-16","2014-08-09","2014-09-02","2014-09-26"]
 ##############################################'FQ19W'  
beam = 'FQ19W'     
dates=["2014-06-05","2014-06-29","2014-07-23","2014-08-16","2014-09-09"] 
##############################################
"""
###################################################################################################################################################
Loading datacube of array_2D_of_coherency matrices
###########################################################################################################################################################
"""
print("Loading datacube...")
root = "D:\\Juanma\\Seville 2014\\"
#T_mult=np.load('X:\\crs2\\Paper2_Agrisar\\T_Stack\\T_stack_asc.npy')
T_mult=np.load(root+beam+"\\T_stack.npy")
print("")
#print("Row "+str(img+1)+", Column "+ str(img_2+1))
print("Processing added scattering mechanisms...")
img = 0
#img_2 = 1
N = T_mult.shape[0]
x_max = T_mult.shape[1]
y_max = T_mult.shape[2]    

save = 1
#outall = "D:\\Juanma\\Seville 2014\\"+beam+"\\Change_detection\\"
outall = root+beam+"\\Change_detection\\Consecutive_changes\\"
application = "Difference change detection" # This is to initialise a class. Other options are: "Single image","ratio change detection".
for img_2 in range(N-1):        
    date = dates[img_2]+' - '+ dates[img_2+1] 
    T11 = T_mult[img_2,:,:,:,:] # img_2 can be = 0 in T11 to do change detection of stack wrt to first image. 
    T22 = T_mult[img_2+1,:,:,:,:] 
    #sar.visRGB_from_T(T_mult[img_2+1,:,:,1,1], T_mult[img_2+1,:,:,2,2], T_mult[img_2+1,:,:,0,0],"",factor=3.5,save=0,outall="")
    
    Tc = (T22 - T11) 
    eigen = sar.eigendecompositions(application)     
    List_RGB = eigen.gral_eigendecomposition(Tc) # to store eigendecomposition results in the class
    R_i,G_i,B_i = eigen.vis(eigen.L1_inc,eigen.L2_inc,eigen.L3_inc,add_or_remove='added') # added SMs
    R_d,G_d,B_d = eigen.vis(eigen.L1_dec,eigen.L2_dec,eigen.L3_dec,add_or_remove='removed') # removed SMs
    
    size = R_i.shape
    a = np.zeros([size[0],size[1],3])
    # Enable or disable denominator to enable or disable normalizing the colours. Makes green stronger.
    a[:,:,0] = R_i#/np.nanmean(R_i) # To plot Added SMs
    a[:,:,1] = G_i#/np.nanmean(G_i)
    a[:,:,2] = B_i#/np.nanmean(B_i)
    b = np.zeros([size[0],size[1],3])
    b[:,:,0] = np.abs(R_d)#/np.abs(np.nanmean(R_d)) #To plot removed SMs
    b[:,:,1] = np.abs(G_d)#/np.abs(np.nanmean(G_d))
    b[:,:,2] = np.abs(B_d)#/np.abs(np.nanmean(B_d))
    
    fig,(ax,ax1)=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,figsize=(15,9))
#    ax.imshow(a*0.15)
#    ax1.imshow(b*0.15)
    ax.imshow(a*3)
    ax1.imshow(b*3)
    ax.axis("off")
    ax1.axis("off")
    plt.tight_layout()
    if save == 1:
        print("saving "+date+" ..." )
        outall1= outall+date+'.png'
        fig.savefig(outall1, bbox_inches='tight',dpi=1200)    
        
#for img2 in range(N-1):
#    sar.visRGB_from_T(T_mult[img2+1,:,:,1,1], T_mult[img2+1,:,:,2,2], T_mult[img2+1,:,:,0,0],"",factor=3.5,save=0,outall="")


