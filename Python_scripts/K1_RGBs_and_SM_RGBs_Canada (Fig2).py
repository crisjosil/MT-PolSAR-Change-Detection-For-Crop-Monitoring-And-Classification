# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 07:40:19 2021

@author: Cristian Silva
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt 
import numpy as np
#import geopandas as gpd
import SAR_utilities_V3a as sar
"""
###########################################################################################################################################################
#################################################################### open image components ############################
###########################################################################################################################################################
"""
x_min=0
y_min=0
datatype = 'float32'
T11_master ="T11_mst_08Sep2009"
folder1 = "D:\\Paper2\\Ascending_only_V1\\Stack_V1.data\\"

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

all_files=os.listdir(folder1)
cols,rows,header=sar.read_config_file_snap(folder1 +"\\"+T11_master)
datatype = 'float32'
basis="P"     # Lexicographic (L) or Pauli (P)
in_format="img" # .Bin
x_min=0
y_min=0
x_max=cols
y_max=rows
ROI_size=[y_min,y_max,x_min,x_max] 

stack_name="T_stack"#"T_stack_asc"
#path_to_save_stack = "D:\\Juanma\\Seville 2014\\"+beam+"\\"+stack_name
path_to_RGB = "D:\\Juanma - Agrisar 2009\\My_results\\RGBs\\No_scale\\"
path_to_SM =  "D:\\Juanma - Agrisar 2009\\My_results\\SMs\\No_scale\\þ"
#date = "2014-08-09"
T_mult=np.zeros((len(dates),(x_max-x_min),(y_max-y_min),3,3),dtype=complex)
save_RGBs=1
N = len(dates2)
for i in range(N):
    date= dates2[i]
    a=[]
    for file in os.listdir(folder1):
        if file.endswith(".img"):
            if dates2[i] in file:
                a.append(file)    
    ROI_size=[y_min,y_max,x_min,x_max]
    T11=sar.array2D_of_coherency_matrices_from_stack_SNAP(folder1,basis,in_format,ROI_size,header,datatype,a)
    T_mult[i,:,:] = T11
    #title = "RGB_"+date
    #sar.visRGB_from_T(T11[:,:,1,1], T11[:,:,2,2], T11[:,:,0,0],title,factor=3.5,save=save_RGBs,outall=path_to_RGB) # enable to scale
    size = T11.shape
    a = np.zeros([size[0],size[1],3])
    a[:,:,0] = np.abs(T11[:,:,1,1])#/np.nanmean(R_i)
    a[:,:,1] = np.abs(T11[:,:,2,2])#/np.nanmean(G_i)
    a[:,:,2] = np.abs(T11[:,:,0,0])#/np.nanmean(B_i)      
    fig,(ax)=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(15,9))
    ax.imshow(a*7)
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
    ax.imshow(a*3.5)
    ax.axis("off")
    plt.tight_layout()
    if save_RGBs == 1:
        print("saving "+date+" ..." )
        outall1= path_to_SM+date+'.png'
        fig.savefig(outall1, bbox_inches='tight',dpi=1200)  

#np.save(path_to_save_stack, T_mult, allow_pickle=True, fix_imports=True)
