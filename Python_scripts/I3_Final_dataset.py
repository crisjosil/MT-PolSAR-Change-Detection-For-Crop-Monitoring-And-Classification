# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:40:49 2020

@author: crs2
"""
########################################################## Imports #################################
#import matplotlib.pyplot as plt 
import numpy as np
#import geopandas as gpd
#import SAR_utilities_V3a as sar
from collections import Counter

class create_datasets():
    def __init__(self,):
        self.a=0
        
    def load_in_arrays( self,
                        path_labels='D:\\Paper2\\Ascending_only_V1\\np_arrays\\Labels_corrected.npy',
                        path_flags=      'D:\\Paper2\\Ascending_only_V1\\np_arrays\\Train_test_flags_asc_Valid.npy',
                        path_CM_datacube='D:\\Paper2\\Ascending_only_V1\\np_arrays\\Change_matrix\\V1\\CM_Datacube_with_diag_viz_alb.npy',
                        path_polSAR_datacube='D:\\Paper2\\Ascending_only_V1\\np_arrays\\datacube_MT_asc.npy',
                        Path_T_stack="D:\Paper2\Ascending_only_V1\\np_arrays\\T_stack_asc.npy"):
        #layer_name = "CM_Datacube.npy"
        #CM_Datacube_with_diag
        #layer_name = "CM_Datacube_with_diag.npy"
        #layer_name = "CM_Datacube_viz_alb.npy"
        #layer_name = "CM_Datacube_with_diag_viz_alb.npy"
        ##############################################
        print("Note: Whishart dataset creation currently disabled")
        
        print("loading ground truth labels array ...")
        self.labels=np.load(path_labels,allow_pickle=True) 
        print("loading train/validation/test flags ...")
        self.tr_ts_arr=np.load(path_flags,  allow_pickle=True) 
        print("loading change matrix cube ...")
        self.CM_Datacube=np.load(path_CM_datacube,  allow_pickle=True) 
        print("loading PolSAR datacube ...")
        self.datacube_MT=np.load(path_polSAR_datacube,  allow_pickle=True)
        #print("loading T_stack to create Whishart based datasets ...")
        #self.T_stack=np.load(Path_T_stack,  allow_pickle=True)
        return

    def call(self):        
        """
        ##################################
        ######## PolSAR features cube: Create train and test sets for ML classification
        ###################################
        """
        N=self.datacube_MT.shape[0]          # Number of images
        n_feat=self.datacube_MT.shape[1]     # Number of features 
        
        # Count number of training and testing pixels
        n_samples_train=0
        n_samples_train_2=0
        n_samples_test=0
        training_samples=0
        testing_samples=0
        training_samples_2=0
        for i in range(self.tr_ts_arr.shape[0]):
            for j in range(self.tr_ts_arr.shape[1]):
                if self.tr_ts_arr[i,j]==0:
                    training_samples=training_samples+1 # train level 1
                elif self.tr_ts_arr[i,j]==1:
                    training_samples_2=training_samples_2+1 # train level 2
                elif self.tr_ts_arr[i,j]==2:
                    testing_samples=testing_samples+1 # Test
        
        # Empty arrays for ML classification    
        X_train = np.zeros([training_samples,N*n_feat])
        X_train_lev2 = np.zeros([training_samples_2,N*n_feat])
        y_train_0 = np.zeros(training_samples)
        y_train_0_lev2 = np.zeros(training_samples_2)
        X_test = np.zeros([testing_samples,N*n_feat])
        y_test = np.zeros(testing_samples)
        
        n_feat_CM = self.CM_Datacube[0,0,:,:,:].flatten().shape[0]
        # Empty arrays for ML classification    
        X_train_CM = np.zeros([training_samples,n_feat_CM])
        X_train_CM_lev2 = np.zeros([training_samples_2,n_feat_CM])
        y_train_CM = np.zeros(training_samples)
        y_train_CM_lev2 = np.zeros(training_samples_2)
        X_test_CM = np.zeros([testing_samples,n_feat_CM])
        y_test_CM = np.zeros(testing_samples)
        
        # Empty arrays for wishart classification    
#        X_train_W= np.zeros([training_samples,N,3,3],dtype=complex)
#        X_train_W_lev2= np.zeros([training_samples_2,N,3,3],dtype=complex)
#        y_train_W = np.zeros(training_samples)
#        y_train_W_lev2 = np.zeros(training_samples_2)
#        X_test_W = np.zeros([testing_samples,N,3,3],dtype=complex)
#        y_test_W = np.zeros(testing_samples)
        # aux variables
        nan_pos_list_train=[]
        nan_pos_list_train_lev2=[]
        nan_pos_list_test=[]
        nan_pos_list_train_CM=[]
        nan_pos_list_train_CM_lev2=[]
        nan_pos_list_test_CM=[]
        nan_pos_list_train_W=[]
        nan_pos_list_train_W_lev2=[]
        nan_pos_list_test_W=[]
        
        n_samples_train=-1
        n_samples_train_2=-1
        n_samples_test=-1
        # Determine if a pixel is training or test and add its corresponding features and crop type label to the empty arrays
        for h in range(self.tr_ts_arr.shape[0]):
            #print(str(int(100*h/tr_ts_arr.shape[0]))+str('%')) # % of progress
            for r in range(self.tr_ts_arr.shape[1]):
                if self.tr_ts_arr[h,r]==0: # is it a training pixel level 1?
                    n_samples_train=n_samples_train+1  # used as index in X_train and y_train_0 array
                    # PolSAR features
                    var=np.abs(self.datacube_MT[:,:,h,r]) # get the features vector per image (N,n_feat)
                    X_train[n_samples_train,:] = var.reshape(var.shape[0]*var.shape[1]) # reshape to 1D (N*n_feat)
                    y_train_0[n_samples_train]   = self.labels[h,r] # get the corresponding class label
                    if np.isnan(np.sum(var))==True:  # if there is a nan in the feature vector, save the position to remove it later
                        nan_pos_list_train.append(n_samples_train)
                    # change matrix
                    var_CM=self.CM_Datacube[h,r,:,:,:].flatten() # flattened change matrix of the pixel h,r
                    X_train_CM[n_samples_train,:] = var_CM
                    y_train_CM[n_samples_train]   = self.labels[h,r] # get the corresponding class label
                    if np.isnan(np.sum(var_CM))==True:  # if there is a nan in the feature vector, save the position to remove it later
                        nan_pos_list_train_CM.append(n_samples_train)
                    # Whishart sequences
#                    var_Wishart=self.T_stack[:,h,r,:,:] # sequence of wishart matrices of the pixel h,r
#                    X_train_W[n_samples_train,:,:,:] = var_Wishart
#                    y_train_W[n_samples_train]   = self.labels[h,r] # get the corresponding class label
#                    if np.isnan(np.sum(var_Wishart))==True:  # if there is a nan in the feature vector, save the position to remove it later
#                        nan_pos_list_train_W.append(n_samples_train)
                    
                elif self.tr_ts_arr[h,r]==1: # is it a level 2 training pixel?
                    n_samples_train_2=n_samples_train_2+1  # used as index in X_train and y_train_0 array
                    # PolSAR features
                    var=np.abs(self.datacube_MT[:,:,h,r]) # get the features vector per image (N,n_feat)
                    X_train_lev2[n_samples_train_2,:] = var.reshape(var.shape[0]*var.shape[1]) # reshape to 1D (N*n_feat)
                    y_train_0_lev2[n_samples_train_2]   = self.labels[h,r] # get the corresponding class label
                    if np.isnan(np.sum(var))==True:  # if there is a nan in the feature vector, save the position to remove it later
                        nan_pos_list_train_lev2.append(n_samples_train_2)
                    # change matrix
                    var_CM=self.CM_Datacube[h,r,:,:,:].flatten() # flattened change matrix of the pixel h,r
                    X_train_CM_lev2[n_samples_train_2,:] = var_CM
                    y_train_CM_lev2[n_samples_train_2]   = self.labels[h,r] # get the corresponding class label
                    if np.isnan(np.sum(var_CM))==True:  # if there is a nan in the feature vector, save the position to remove it later
                        nan_pos_list_train_CM_lev2.append(n_samples_train_2)
                    # Whishart sequences
#                    var_Wishart=self.T_stack[:,h,r,:,:] # sequence of wishart matrices of the pixel h,r
#                    X_train_W_lev2[n_samples_train_2,:,:,:] = var_Wishart
#                    y_train_W_lev2[n_samples_train_2]   = self.labels[h,r] # get the corresponding class label
#                    if np.isnan(np.sum(var_Wishart))==True:  # if there is a nan in the feature vector, save the position to remove it later
#                        nan_pos_list_train_W_lev2.append(n_samples_train_2)            
                    
                    
                elif self.tr_ts_arr[h,r]==2: # is it a testing pixel?
                    n_samples_test=n_samples_test+1
                    # PolSAR features
                    var=np.abs(self.datacube_MT[:,:,h,r])
                    X_test[n_samples_test,:] = var.reshape(var.shape[0]*var.shape[1])
                    y_test[n_samples_test] = self.labels[h,r]
                    if np.isnan(np.sum(var))==True:
                        nan_pos_list_test.append(n_samples_test)
                    # Change matrix    
                    var_CM=self.CM_Datacube[h,r,:,:,:].flatten() # flattened change matrix of the pixel h,r
                    X_test_CM[n_samples_test,:] = var_CM
                    y_test_CM[n_samples_test] = self.labels[h,r]
                    if np.isnan(np.sum(var_CM))==True:
                        nan_pos_list_test_CM.append(n_samples_test)
                    # Whishart sequences   
#                    var_Wishart=self.T_stack[:,h,r,:,:] # sequence of wishart matrices of the pixel h,r
#                    X_test_W[n_samples_test,:,:,:] = var_Wishart
#                    y_test_W[n_samples_test] = self.labels[h,r]
#                    if np.isnan(np.sum(var_Wishart))==True:
#                        nan_pos_list_test_W.append(n_samples_test)
                    
        print("Train:")
        print(X_train.shape)
        print(X_train_CM.shape)
        #print(X_train_W.shape)
        print(y_train_0.shape)
        print(y_train_CM.shape)
        #print(y_train_W.shape)
        print(" ")
        print("Validation:")
        print(X_train_lev2.shape)
        print(X_train_CM_lev2.shape)
        #print(X_train_W_lev2.shape)
        print(y_train_0_lev2.shape)
        print(y_train_CM_lev2.shape)
        #print(y_train_W_lev2.shape)
        print(" ")
        print("Test:")
        print(X_test.shape)
        print(X_test_CM.shape)
        #print(X_test_W.shape)
        print(y_test.shape)
        print(y_test_CM.shape)
        #print(y_test_W.shape)
        print(" ")
        print("nan_pos_list_train:")
        print(len(nan_pos_list_train))
        print(len(nan_pos_list_train_CM))
        print(len(nan_pos_list_train_W))
        print(" ")
        print("nan_pos_list_train_lev2:")
        print(len(nan_pos_list_train_lev2))
        print(len(nan_pos_list_train_CM_lev2))
        print(len(nan_pos_list_train_W_lev2))
        print(" ")
        print("nan_pos_list_test:")
        print(len(nan_pos_list_test))
        print(len(nan_pos_list_test_CM))
        print(len(nan_pos_list_test_W))
                        
        # Delete the rows with nans                
        X_train2=np.delete(X_train,   nan_pos_list_train,axis=0)
        y_train2=np.delete(y_train_0, nan_pos_list_train,axis=0)
        X_val2=np.delete(X_train_lev2,   nan_pos_list_train_lev2,axis=0)
        y_val2=np.delete(y_train_0_lev2, nan_pos_list_train_lev2,axis=0)
        X_test2=np.delete(X_test, nan_pos_list_test,axis=0)
        y_test2=np.delete(y_test, nan_pos_list_test,axis=0)
        print(Counter(y_train2))
        print(Counter(y_val2))
        print(Counter(y_test2))
        self.X_train2=X_train2
        self.y_train2=y_train2
        self.X_val2=X_val2
        self.y_val2=y_val2
        self.X_test2=X_test2
        self.y_test2=y_test2

         # Delete the rows with nans                
        X_train_CM2=np.delete(X_train_CM, nan_pos_list_train,axis=0)
        y_train_CM2=np.delete(y_train_CM, nan_pos_list_train,axis=0)
        X_val2_CM2=np.delete(X_train_CM_lev2, nan_pos_list_train_lev2,axis=0)
        y_val2_CM2=np.delete(y_train_CM_lev2, nan_pos_list_train_lev2,axis=0)
        X_test_CM2=np.delete(X_test_CM, nan_pos_list_test,axis=0)
        y_test_CM2=np.delete(y_test_CM, nan_pos_list_test,axis=0)
        print(Counter(y_train_CM2))
        print(Counter(y_val2_CM2))
        print(Counter(y_test_CM2))
        self.X_train_CM2=X_train_CM2
        self.y_train_CM2=y_train_CM2
        self.X_val2_CM2=X_val2_CM2
        self.y_val2_CM2=y_val2_CM2
        self.X_test_CM2=X_test_CM2
        self.y_test_CM2=y_test_CM2
         # Delete the rows with nans                
#        X_train_W2=np.delete(X_train_W, nan_pos_list_train,axis=0)
#        y_train_W2=np.delete(y_train_W, nan_pos_list_train,axis=0)
#        X_val_W2=np.delete(X_train_W_lev2, nan_pos_list_train_lev2,axis=0)
#        y_val_W2=np.delete(y_train_W_lev2, nan_pos_list_train_lev2,axis=0)
#        
#        X_test_W2=np.delete(X_test_W, nan_pos_list_test,axis=0)
#        y_test_W2=np.delete(y_test_W, nan_pos_list_test,axis=0)
        #print(Counter(y_train_W2))
        #print(Counter(y_val_W2))

    def save_arrays(self, save_PolSAR=True,save_CM=True,
                    location_PolSAR="D:\\Paper2\\Ascending_only_V1\\np_arrays\\Final_datasets\\PolSAR_F\\V2\\",
                    location_CM="D:\\Paper2\\Ascending_only_V1\\np_arrays\\Final_datasets\\CM\\V2\\",
                    PolSAR_arr_names=["X_train","y_train","X_val","y_val","X_test","y_test"],
                    CM_arr_names=["X_train","y_train","X_val","y_val","X_test","y_test"]):
        
        if save_PolSAR == True:
            layer_name = PolSAR_arr_names[0] #
            np.save(location_PolSAR+layer_name, self.X_train2, allow_pickle=True, fix_imports=True)
            layer_name = PolSAR_arr_names[1]
            np.save(location_PolSAR+layer_name, self.y_train2, allow_pickle=True, fix_imports=True)
            layer_name = PolSAR_arr_names[2] #
            np.save(location_PolSAR+layer_name, self.X_val2, allow_pickle=True, fix_imports=True)
            layer_name = PolSAR_arr_names[3]
            np.save(location_PolSAR+layer_name, self.y_val2, allow_pickle=True, fix_imports=True)
            layer_name = PolSAR_arr_names[4]
            np.save(location_PolSAR+layer_name, self.X_test2, allow_pickle=True, fix_imports=True)
            layer_name = PolSAR_arr_names[5]
            np.save(location_PolSAR+layer_name, self.y_test2, allow_pickle=True, fix_imports=True)
        
        if save_CM == True:
            layer_name = CM_arr_names[0] #CM_Datacube_viz_alb #X_train_viz_alb_with_diag
            np.save(location_CM+layer_name, self.X_train_CM2, allow_pickle=True, fix_imports=True)
            layer_name = CM_arr_names[1]
            np.save(location_CM+layer_name, self.y_train_CM2, allow_pickle=True, fix_imports=True)
            layer_name = CM_arr_names[2] #CM_Datacube_viz_alb
            np.save(location_CM+layer_name, self.X_val2_CM2, allow_pickle=True, fix_imports=True)
            layer_name = CM_arr_names[3]
            np.save(location_CM+layer_name, self.y_val2_CM2, allow_pickle=True, fix_imports=True)
            layer_name = CM_arr_names[4]
            np.save(location_CM+layer_name, self.X_test_CM2, allow_pickle=True, fix_imports=True)
            layer_name = CM_arr_names[5]
            np.save(location_CM+layer_name, self.y_test_CM2, allow_pickle=True, fix_imports=True)
        
        #location="D:\\Paper2\\Ascending_only_V1\\np_arrays\\Final_datasets\\Wishart\\V1\\"
        #layer_name = "X_train" #
        #np.save(location+layer_name, X_train_W2, allow_pickle=True, fix_imports=True)
        #layer_name = "y_train"
        #np.save(location+layer_name, y_train_W, allow_pickle=True, fix_imports=True)
        #layer_name = "X_test"
        #np.save(location+layer_name, X_test_W2, allow_pickle=True, fix_imports=True)
        #layer_name = "y_test"
        #np.save(location+layer_name, y_test_W, allow_pickle=True, fix_imports=True)
        

#final_datasets=create_datasets()
#final_datasets.load_in_arrays() # change default paths if needed
#final_datasets.call() # execute operations
#final_datasets.save_arrays() # change default paths if needed