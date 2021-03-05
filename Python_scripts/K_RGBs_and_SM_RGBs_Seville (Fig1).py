# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 07:50:51 2021

@author: crs2
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt 
import numpy as np
#import geopandas as gpd
import SAR_utilities_V3a as sar
 ##############################################'FQ13W'  
beam = 'FQ8W'     
dates=["2014-05-22","2014-06-15","2014-07-09","2014-08-02","2014-08-26","2014-09-19"]
##############################################'FQ13W'  
#beam = 'FQ13W'     
#dates=["2014-06-22","2014-07-16","2014-08-09","2014-09-02","2014-09-26"]
 ##############################################'FQ13W'  
#beam = 'FQ19W'     
#dates=["2014-06-05","2014-06-29","2014-07-23","2014-08-16","2014-09-09"] 



folder="D:\\Juanma\\Seville 2014\\"+beam+"\\"+dates[0]+".rds2\\" #files[img] # path of the image
stack_name="T_stack"#"T_stack_asc"
path_to_save_stack = "D:\\Juanma\\Seville 2014\\"+beam+"\\"+stack_name
path_to_RGB = "D:\\Juanma\\Seville 2014\\"+beam+"\\RGBs\\No_scale\\"
path_to_SM =  "D:\\Juanma\\Seville 2014\\"+beam+"\\SMs\\No_scale\\"

cols,rows,header=sar.read_config_file(folder)
datatype = 'float32'
basis="L"     # Lexicographic (L) or Pauli (P)
in_format="Bin" # .Bin
x_min=2000
y_min=2000
x_max=cols -500
y_max=rows - 500
ROI_size=[y_min,y_max,x_min,x_max] 


#date = "2014-08-09"
T_mult=np.zeros((len(dates),(x_max-x_min),(y_max-y_min),3,3),dtype=complex)
save_RGBs=1
N = len(dates)
for i in range(N):
    date=dates[i]
    folder = "D:\\Juanma\\Seville 2014\\"+beam+"\\"+date+".rds2\\"
    T11=sar.array2D_of_coherency_matrices(folder,basis,in_format,ROI_size,header,datatype)
    T_mult[i,:,:] = T11
    #title = "RGB_"+date
    #sar.visRGB_from_T(T11[:,:,1,1], T11[:,:,2,2], T11[:,:,0,0],title,factor=3.5,save=save_RGBs,outall=path_to_RGB) # enable to scale
    size = T11.shape
    a = np.zeros([size[0],size[1],3])
    a[:,:,0] = np.abs(T11[:,:,1,1])#/np.nanmean(R_i)
    a[:,:,1] = np.abs(T11[:,:,2,2])#/np.nanmean(G_i)
    a[:,:,2] = np.abs(T11[:,:,0,0])#/np.nanmean(B_i)      
    fig,(ax)=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(15,9))
    ax.imshow(a*2)
    ax.axis("off")
    plt.tight_layout()
    if save_RGBs == 1:
        print("saving "+date+" ..." )
        outall1= path_to_RGB+date+'.png'
        fig.savefig(outall1, bbox_inches='tight',dpi=1200)  
        
    application = "Single image"
    eigen = sar.eigendecompositions(application)
    eigen.gral_eigendecomposition(T11) # Executes function to store eigendecomposition results in the class
    R,G,B = eigen.vis(eigen.L1,eigen.L2,eigen.L3,add_or_remove='added') # added is same as dominant 
    
    # Dominant PolSAR Scattering mechanism
    #title = "SM_"+date
    #sar.visRGB_from_T(R, G, B,title,factor=2.5,save=save_RGBs,outall=path_to_SM) # enable to scale
    size = R.shape
    a = np.zeros([size[0],size[1],3])
    a[:,:,0] = R#/np.nanmean(R_i)
    a[:,:,1] = G#/np.nanmean(G_i)
    a[:,:,2] = B#/np.nanmean(B_i)
    
    fig,(ax)=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(15,9))
    ax.imshow(a*2)
    ax.axis("off")
    plt.tight_layout()
    if save_RGBs == 1:
        print("saving "+date+" ..." )
        outall1= path_to_SM+date+'.png'
        fig.savefig(outall1, bbox_inches='tight',dpi=1200)  

#np.save(path_to_save_stack, T_mult, allow_pickle=True, fix_imports=True)

