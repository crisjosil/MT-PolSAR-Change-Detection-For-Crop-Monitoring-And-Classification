# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 12:21:49 2020
This scripts generates a multitemporal datacube of PolSAR features
It is an array with shape [N_images, N_feaures,dr,da] where:
    N_images: Number of images
    N_feature: Number of features

Current features are:
    feat_list=['VV','VH','HH','VH_VV','HH_VV','L1','alpha1','beta1','L_avg','alpha_avg','entropy','anisotropy'] 
@author: Cristian Silva
"""
########################################################## Imports #################################
import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as ln
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal  
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import sys
#sys.path.insert(1, 'D:\\Juanma - Agrisar 2009\\Full_data\\Python\\Temporal Change detection\\')
import SAR_utilities_V3a as sar

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
##################################
######## Seville or San Francisco
##################################
#folder="D:\\Datasets\\SanFrancisco_C3_Bin\\"
#folder="D:\\Juanma\Seville 2014\\FQ13W\\2014-08-09.rds2\\"
#x_max,y_max,header=sar.read_config_file(folder)
#basis="L"     # Lexicographic (L) or Pauli (P)
#in_format="Bin" # Bin or img
#y_min=3000
#y_max=6200
#x_min=3500
#x_max=6500
#################################
######## AgriSAR
#################################
#folder="X:\\crs2\\Paper2_Agrisar\\tests\\A\\RS2_OK6154_PK78632_DK76410_FQ2_20090721_132229_HH_VV_HV_VH_SLC.data\\"
#x_max,y_max,header=sar.read_config_file_snap(folder+"T11")
"""
###########################################################################################################################################################
Creating the 2D_array_of_choherency_matrices
###########################################################################################################################################################
"""
#root="X:\\crs2\\Paper2_Agrisar\\From_June_to_Sept\\RS2_OK5763_PK74503_DK72509_FQ2_20090608_002856_HH_VV_HV_VH_SLC_Stack_asc_desc.data\\"
#folders=[    #root+"2009-06-03 - Copy\\",
#             root+"2009-06-04 - Copy\\",
#             root+"2009-06-08 - Copy\\",
#             root+"2009-06-10 - Copy\\", 
#             root+"2009-06-24 - Copy\\",
#             #root+"2009-07-02 - Copy\\",
#             root+"2009-07-04 - Copy\\",
#             root+"2009-07-11 - Copy\\",
#             root+"2009-07-21 - Copy\\",
#             root+"2009-07-26 - Copy\\",
#             root+"2009-08-04 - Copy\\",
#             root+"2009-08-11 - Copy\\",
#             #root+"2009-08-12 - Copy\\",
#             root+"2009-08-19 - Copy\\",
#             root+"2009-08-22 - Copy\\",
#             root+"2009-09-08 - Copy\\"]
##folder="X:\\crs2\\Paper2_Agrisar\\From_June_to_Sept\\RS2_OK5763_PK74503_DK72509_FQ2_20090608_002856_HH_VV_HV_VH_SLC_Stack_asc_desc.data\\2009-06-10 - Copy\\" #2009-06-08 - Copy\\"
#x_max,y_max,header=sar.read_config_file_snap(folders[0]+"T11")
#basis="P"     # Lexicographic (L) or Pauli (P)
#in_format="img" # Bin or img
#N = len(folders)
#feat_list=['VV','VH','HH','VH_VV','HH_VV','L1','alpha1','beta1','L_avg','alpha_avg','entropy','anisotropy']
#datacube_MT=np.zeros([N,len(feat_list),x_max,y_max],dtype=np.complex64)
#################################
############### Crop the image to some known size otherwise comment (disable) the lines below
#################################
#ROI_size=[y_min,y_max,x_min,x_max]
# Compute image size
#da=(x_max-x_min)
#dr=(y_max-y_min) 
"""
###################################################################################################################################################
Loading datacube of array_2D_of_coherency matrices
###########################################################################################################################################################
"""
print("Loading datacube...")
#T_mult=np.load('X:\\crs2\\Paper2_Agrisar\\T_Stack\\T_stack_asc.npy')
T_mult=np.load('D:\Paper2\Ascending_only_V1\\np_arrays\\T_stack_asc.npy')
#########################################################################################################

feat_list=['VV','VH','HH','VH_VV','HH_VV','HH_VH','L1','alpha1','beta1','L_avg','alpha_avg','entropy','anisotropy','R_avg','G_avg','B_avg'] # repeated
x_max = T_mult.shape[1]
y_max = T_mult.shape[2]
N = T_mult.shape[0]
datacube_MT=np.zeros([N,len(feat_list),x_max,y_max],dtype=np.complex64)
for i in range(N):
    print("Img "+str(i+1)+" of "+str(N))
    #folder = folders[i]
    """  
    ###########################################################################################################################################################
    # Read the covariance matrix for each resolution cell                      ############################
    ###########################################################################################################################################################
    """
    T=T_mult[i,:,:,:,:]    
    #T=sar.array2D_of_coherency_matrices(folder,basis,in_format,ROI_size,header,datatype)
    # show RGB
    #sar.visRGB_from_T(T[:,:,1,1], T[:,:,2,2], T[:,:,0,0],"",factor=2.5)
    """  
    ###########################################################################################################################################################
    # return the eigenvalue and eigenvector decomposition. 2D array of image size, in which each position contains 3 eigenvelues, or 3x3 eigenvectors #########
    ###########################################################################################################################################################
    """
    L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33=sar.eigendecomposition_vectorized(T)
    
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
    datacube_1[0,:,:]= T[:,:,0,0] # VV
    datacube_1[1,:,:]= T[:,:,1,1] # VH
    datacube_1[2,:,:]= T[:,:,2,2] # HH
    datacube_1[3,:,:]= T[:,:,1,1]/T[:,:,0,0] # VH/VV
    datacube_1[4,:,:]= T[:,:,2,2]/T[:,:,0,0] # HH/VV
    datacube_1[5,:,:]= T[:,:,2,2]/T[:,:,1,1] # HH/VH
    datacube_1[6,:,:]= L1
    datacube_1[7,:,:]= np.arccos(U_11) #alpha1
    datacube_1[8,:,:]= np.arccos((U_21)/np.sin(np.arccos(U_11))) #beta1
    datacube_1[9,:,:]= L_avg
    datacube_1[10,:,:]= alpha_avg
    datacube_1[11,:,:]= entropy
    datacube_1[12,:,:]= anisotropy
    datacube_1[13,:,:]= R_avg
    datacube_1[14,:,:]= G_avg
    datacube_1[15,:,:]= B_avg
    # Save the PolSAR features of all images
    datacube_MT[i,:,:,:]=datacube_1

plt.close('all')
del datacube_1
####################################################################
###################### Plot multitemporal maps of a feature 
####################################################################    
vmin=0
vmax=1    
for i in range(datacube_MT.shape[0]):
    title=" Entropy "+str(i)
    #Entropy_avg=sar.plot_descriptor(datacube_MT[i,10,:,:],vmin,vmax,title)
    descriptor=np.abs(datacube_MT[i,10,:,:])
    fig, (ax10) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax10.set_title(title)
    im10=ax10.imshow(descriptor, cmap = 'jet', vmin=vmin, vmax=vmax)
    ax10.axis('off')
    sar.color_bar(im10) # 
    plt.tight_layout()  


"""
Save the multitemporal datacube of PolSAR features
"""
# save
layer_name = "datacube_MT_asc"
np.save("D:\\Paper2\\Ascending_only_V1\\np_arrays\\"+layer_name, datacube_MT, allow_pickle=True, fix_imports=True)

imgg=6
# RGB
sar.visRGB_from_T(datacube_MT[imgg,1,:,:], datacube_MT[imgg,2,:,:], datacube_MT[imgg,0,:,:],"",factor=3.5)
# Dominant PolSAR Scattering mechanism
sar.visRGB_from_T(datacube_MT[imgg,13,:,:], datacube_MT[imgg,14,:,:], datacube_MT[imgg,15,:,:],"",factor=3.5)