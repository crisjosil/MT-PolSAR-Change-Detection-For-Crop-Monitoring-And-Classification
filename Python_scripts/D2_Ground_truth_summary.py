# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:17:14 2020
This script generates a summary of the ground truth data once cropped to match the
stack of coregistered images.
- First read the ground truth, IDs and area layers previously saved with the np_Labels_from_shp_V0.py script
- Then for every crop type, find the unique parcel IDs with the corresponding area and the associated number of pixels
- Then store the results in summary dataframes

Split 75% of parcels for train (level 1 (70%) and 2 (30%)) and 25% for test parcels. 
Thus corresponds to approximately 52.5% for training level 1, 22.5% for training level 2 and 25% for test.

The dictionary "d" contains detailed information about the specific parcels and pixels seleccted for training/test
@author: crs2
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt 
import numpy as np
import os
#os.chdir('D:\\Juanma - Agrisar 2009\\Full_data\\Python\\Temporal Change detection')
import SAR_utilities_V3a as sar
import fiona
import pandas as pd
from collections import Counter
import random
import matplotlib.patches as mpatches
################################################# read shp

class split_ground_truth():
    """
    Split parcels into training, validation, test. Note that the splitting is by parcels which for SAR is recommended rather than by pixels
    split_train ( float between 0 to 1): % of parcels where pixels will be for training
    split_validation ( float between 0 to 1): % of parcels where pixels will be for validation
    The test split is calculated as: 1 - (split_train + split_validation)
    
    Note that the validation split can also be used as level 2 training (ensemble) since the data are split by parcels and not by pixels
    """
    def __init__(self,split_train=0.6,split_validation=0.2,split_test=0.2):
        self.split_train = split_train
        self.split_validation = split_validation
        self.split_test = split_test
        self.split_test = 1 - (split_train + split_validation)

    def load_arrays(self,
                    path_shp="D:\\Paper2\\Ascending_only_V1\\AgriSAR2009.shp",
                    path_labels = 'D:\\Paper2\\Ascending_only_V1\\np_arrays\\Labels_asc.npy',
                    path_Parcel_IDs = 'D:\\Paper2\\Ascending_only_V1\\np_arrays\\IDs_asc.npy',
                    path_areas = 'D:\\Paper2\\Ascending_only_V1\\np_arrays\\Areas_asc.npy'):
    
        # Read shapefile with fiona package
        print("Loading and processing shapefile ...")
        Polygons_in_shp=fiona.open(path_shp, "r")
        #read the polygon names and save them in a list
        pol_names=[]
        pol_crops=[]
        for poly in range(len(Polygons_in_shp)):
            pol_names.append(Polygons_in_shp[poly]['properties']['IDENT']  )  
            pol_crops.append(Polygons_in_shp[poly]['properties']['CROP_TYPE']  )  
    
        # initializing lists 
        self.crop_types = list(set(pol_crops))
        self.crop_types.sort()
        crop_types_code = np.arange(0,len( self.crop_types))
          # using dictionary comprehension to convert lists to dictionary 
        self.crop_types_dict_code = {self.crop_types[i]: crop_types_code[i] for i in range(len(self.crop_types))} 
        
        print("Loading and processing labels ...")
        self.labels=np.load(path_labels,allow_pickle=True) # load ground truth as in the stack
        print("Loading and processing parcel IDs ...")
        ids=np.load(path_Parcel_IDs,allow_pickle=True)     # load IDs as in the stack
        self.ids_str = ids.astype('U')
        print("Loading and processing parcel areas ...")
        self.areas=np.load(path_areas,  allow_pickle=True) # load areas as in the stack
        return
        #return(labels,ids_str,areas,crop_types_dict_code)
    
    def uniqueIndexes(l):
        # copied from a forum
        seen = set()
        res = []
        for i, n in enumerate(l):
            if n not in seen:
                res.append(i)
                seen.add(n)
        return res

    def call(self):
        sum_areas_list=[]
        Num_parcels_list=[]
        Num_pixels_list=[]
        parcel_name_list=[]
        pixels_per_parcel_list=[]
        parcels_per_type=[]
        parcel_area_list=[]
        for crop_class in range(len(self.crop_types)):
            print(self.crop_types[crop_class])
            # select pixels for a single crop type
            crop_temp=np.where(self.labels==crop_class,100,np.nan) 
            # filter Ids by crop type
            ids_str_1=np.where(crop_temp==100,self.ids_str,-1)
            areas1=   np.where(crop_temp==100,self.areas,-1)
            # reshape as 1D array
            ids_str_2=np.reshape(ids_str_1,(self.labels.shape[0]*self.labels.shape[1]))
            areas2=   np.reshape(areas1,   (self.labels.shape[0]*self.labels.shape[1]))
            
            ids_str_3 = ids_str_2.tolist() # to list
            areas3 = areas2.tolist()
            ids_str_4 = []
            areas4 = []
            #pos=[]
            # cropland / No cropland
            for i in range(len(ids_str_3)): 
                if ids_str_3[i]!= '-1':
                    ids_str_4.append(ids_str_3[i])# croplands
        
            for i in range(len(areas3)):
                if areas3[i]!= -1:
                    areas4.append(areas3[i])# croplands
        
            pos_of_unique=split_ground_truth.uniqueIndexes(ids_str_4)
            
            parcels_present=[]
            areas_present=[]
            for i in pos_of_unique:
                parcels_present.append(ids_str_4[i])
                areas_present.append(areas4[i])
                
            # unique parcel and areas per crop type          
            Num_pixels=len(ids_str_4)
            #print("Number of pixels: "+str(Num_pixels))
            sum_areas_list.append(np.array(areas_present).sum())
            Num_parcels_list.append(len(parcels_present))
            Num_pixels_list.append(Num_pixels)
            parcel_name_list.append(parcels_present)
            parcel_area_list.append(areas_present)
            pixels_per_parcel_list.append(list(Counter(ids_str_4).values()))
            parcels_per_type.append([self.crop_types[crop_class]]*len(parcels_present))
        
        summary=pd.DataFrame(columns=['Crop_type','Num_parcels','Num_pixels','Area(Ha)'])
        summary['Crop_type']=self.crop_types
        summary['Num_parcels']=Num_parcels_list
        summary['Num_pixels'] =Num_pixels_list
        summary['Area(Ha)']  =sum_areas_list
        summary.set_index('Crop_type',inplace=True)
        
        fig,(ax,ax1,ax2)=plt.subplots(3,sharex=True,figsize=(10,8))
        summary['Num_pixels'].plot.bar(ax=ax)
        summary['Num_parcels'].plot.bar(ax=ax1)
        summary['Area(Ha)'].plot.bar(ax=ax2)
        ax.set_ylabel('Num_pixels')
        ax1.set_ylabel('Num_parcels')
        ax2.set_ylabel('Area(Ha)')
        plt.tight_layout()
        
        fig,(ax)=plt.subplots(1,sharex=True)
        ax.table(cellText=summary.values, colLabels=summary.columns, loc='center')
        
        parcels_per_type_for_df=[]
        parcel_name_list_for_df=[]
        pixels_per_parcel_list_for_df=[]
        parcel_area_list_for_df=[]
        for j in range(len(parcels_per_type)):
            a=parcels_per_type[j]
            b=parcel_name_list[j]
            c=pixels_per_parcel_list[j]
            d=parcel_area_list[j]
            for k in range(len(a)):
                parcels_per_type_for_df.append(a[k])
                parcel_name_list_for_df.append(b[k])
                pixels_per_parcel_list_for_df.append(c[k])
                parcel_area_list_for_df.append(d[k])
                
        summary_by_crop=pd.DataFrame(columns=['Crop_type','ID','Area (Ha)','Num_pixels'])
        summary_by_crop['Crop_type']=parcels_per_type_for_df
        summary_by_crop['ID']=parcel_name_list_for_df
        summary_by_crop['Num_pixels']=pixels_per_parcel_list_for_df    
        summary_by_crop['Area (Ha)']=parcel_area_list_for_df      
        summary_by_crop.set_index('Crop_type',inplace=True) 
        
        #excel_name="summary_asc.xlsx"
        #summary.to_excel("X:\\crs2\\Paper2_Agrisar\\T_Stack\\"+excel_name,sheet_name='Summary')
        #excel_name="summary_by_crop_asc.xlsx"
        #summary_by_crop.to_excel("X:\\crs2\\Paper2_Agrisar\\T_Stack\\"+excel_name,sheet_name='Summary_by_crop')
        #################################### Train / test split ###############################################################
        ############## Save in a nested dictionary the for training and testing. The sum of training pixels accounts for ###########
        ######################## approx 65% of the total pixels in each crop type ###############################
        #########################################################################################################
        # remove crop types in which we dont have enough data to train and test
        print(self.crop_types)
        self.crop_types.remove('Alfalfa')  # class 0
        self.crop_types.remove('Rye (fall)') # class 13
        self.crop_types.remove('Chem-fallow') # class 4
        self.crop_types.remove('Summerfallow')  # class 15
        
        self.ids_str=np.where(self.labels==0, np.nan,self.ids_str) # alfalfa
        self.ids_str=np.where(self.labels==13,np.nan,self.ids_str) # Rye
        self.ids_str=np.where(self.labels==4, np.nan,self.ids_str) # Chem-fallow
        self.ids_str=np.where(self.labels==15,np.nan,self.ids_str) # Summerfallow
        
        # Covert Grass and Mixed Hay to Mixed Pasture
        summary_by_crop['Old_crops'] = summary_by_crop.index
        summary_by_crop['Old_crops'] = summary_by_crop['Old_crops'].str.replace('Grass','Mixed Pasture')
        summary_by_crop['Old_crops'] = summary_by_crop['Old_crops'].str.replace('Mixed Hay','Mixed Pasture')
        summary_by_crop.set_index('Old_crops',inplace=True)
        self.crop_types.remove('Grass')  # class 8
        self.crop_types.remove('Mixed Hay')  # class 10
        
        labels_=self.labels.copy()
        labels_=np.where(labels_==0,np.nan,labels_) # remove alfalfa
        labels_=np.where(labels_==1,0,labels_)
        labels_=np.where(labels_==2,1,labels_)
        labels_=np.where(labels_==3,2,labels_)
        labels_=np.where(labels_==4,np.nan,labels_) # remove Chem-fallow
        labels_=np.where(labels_==5,3,labels_)
        labels_=np.where(labels_==6,4,labels_)
        labels_=np.where(labels_==7,5,labels_)
        labels_=np.where(labels_==8,6,labels_)  # Grass to Mixed pasture
        labels_=np.where(labels_==9,7,labels_)
        labels_=np.where(labels_==10,6,labels_) # Mixed pasture
        labels_=np.where(labels_==11,6,labels_) # Mixed pasture
        labels_=np.where(labels_==12,8,labels_) 
        labels_=np.where(labels_==13,np.nan,labels_)# remove Rye
        labels_=np.where(labels_==14,9,labels_)
        labels_=np.where(labels_==15,np.nan,labels_) # remove Summer-fallow
        self.labels_=labels_
        
        d = {}
        whole_training_IDs=[]
        whole_training_crops=[]
        whole_testing_IDs=[]
        whole_parcel_IDs=[]
        whole_test_crops=[]
        #random.seed(a=42)
        """
        First divide into training and test, combining the percentage of training + validation as training.
        Then subdivide the big training into training and validation 
        Can be definitely improved!
        """
        for crop_label in self.crop_types: # for every crop except alfalfa, Rye, chem f
            #crop_label="Oat"
            crop_temp=summary_by_crop.loc[crop_label]  #filter by crop type
            total_pixels_temp = summary_by_crop.loc[crop_label]['Num_pixels'].values.sum() # Total pixels per class
            training_pixels = total_pixels_temp*(self.split_train + self.split_validation)  # 50% of all pixels in this class     # pixels for training  level 1 (base clssifiers)
            
            training_IDs=[]
            parcels=list(summary_by_crop.loc[crop_label]['ID'].values) # Parcel IDs of this crop type
            whole_parcel_IDs.append(summary_by_crop.loc[crop_label]['ID'].values.tolist()) # append them for future use
            num_pixels = 0 # var to count the pixels in training. Increases incrementally in the while loop
            while num_pixels < training_pixels: #while the accumulated number of pixels is less that 60% of total pixels do
                random_parcel=random.sample(parcels, 1)  # randomly select a parcel     
                pixels_this_parcel=crop_temp[crop_temp['ID']==random_parcel[0]]['Num_pixels'].values[0]# number of pixels in the random parcel
                num_pixels = num_pixels + pixels_this_parcel # aggregate the already existent pixels + the ones from the selected parcel
                training_IDs.append(random_parcel[0]) # append the parcel name
                parcels.remove(random_parcel[0])      # remove it from original list to avoid repetitions. Repeat.
                whole_training_IDs.append(random_parcel[0]) 
                whole_training_crops.append(crop_label)
                
            print("Crop: "+crop_label)
            parcels=list(summary_by_crop.loc[crop_label]['ID'].values)
            print("Training Parcels: "+str(len(training_IDs))+" of "+str(len(parcels)))
            print("Pixels for training: "+str(num_pixels)+" of "+str(total_pixels_temp))
            
            
            # Get name of parcels for Testing: Parcels remaining. 
            testing_IDs = [ tid for tid in parcels if tid not in training_IDs ]
            whole_testing_IDs.append(testing_IDs) 
            whole_test_crops.append([crop_label]*len(testing_IDs))
            print("Testing Parcels: "+str(len(testing_IDs))+" of "+str(len(parcels)))
            print("Pixels for testing: "+str(total_pixels_temp - num_pixels)+" of "+str(total_pixels_temp))
            
            # Nested Dictionary to save results
            
            d[crop_label] = {}
            d[crop_label]['Crop_type'] = crop_label
            d[crop_label]['Total_parcels'] = len(parcels)
            d[crop_label]['Total_pixels'] = total_pixels_temp
            d[crop_label]['Total_training_parcels'] = len(training_IDs)
            d[crop_label]['Total_training_pixels'] = num_pixels
            d[crop_label]['Training_IDs'] = training_IDs
            d[crop_label]['Total_testing_parcels'] = len(testing_IDs)
            d[crop_label]['Total_testing_pixels'] = total_pixels_temp - num_pixels
            d[crop_label]['Perc_testing_pixels'] = (total_pixels_temp - num_pixels)/total_pixels_temp
            d[crop_label]['Testing_IDs'] = testing_IDs
        
        whole_testing_IDs_flat = [item for sublist in whole_testing_IDs for item in sublist] # flat nested list
        #whole_parcel_IDs_flat = [item1 for sublist1 in whole_parcel_IDs for item1 in sublist1]# flat nested list
        whole_test_crops_flat = [item2 for sublist2 in whole_test_crops for item2 in sublist2]# flat nested list
        # create a dataframe that contains all the parcel IDs with its training and testing flag
        tr_df=pd.DataFrame(columns=['Crop','ID','Flag'])
        tr_df1=pd.DataFrame(columns=['Crop','ID','Flag'])
        ts_df=pd.DataFrame(columns=['ID','Flag'])
        tr_df['ID']=whole_training_IDs
        tr_df['Crop']=whole_training_crops
        
        ### Split training data into training and validation
        # Create flags to know if a parcel corresponds to train, validation or test
        for crop_label in self.crop_types:
            ooo = tr_df[tr_df['Crop']==crop_label]
            to_train = int(tr_df[tr_df['Crop']==crop_label].shape[0]*(1-(self.split_validation/(self.split_train+self.split_validation))))
            zeros=np.zeros(to_train)
            ones=np.ones((int(tr_df[tr_df['Crop']==crop_label].shape[0])-to_train))
            ooo['Flag'].iloc[:to_train]=zeros # flag for training 
            ooo['Flag'].iloc[to_train:]=ones  # flag for validation
            tr_df1=pd.concat((tr_df1,ooo),axis=0)
        #tr_df['Flag']=0#'Train'
    
        ts_df['ID']=whole_testing_IDs_flat
        ts_df['Flag']=2# flag for Test pixels
        ts_df['Crop']=whole_test_crops_flat
        tr_ts_df=pd.concat((tr_df1,ts_df),axis=0)
        self.tr_ts_df = tr_ts_df
        
        # Create empty image where to assign the train or test flag
        tr_ts_arr   = np.empty_like(self.areas)
        tr_ts_arr[:] = np.nan
        
        # fill it 
        for k in range(self.ids_str.shape[0]):
            print(str(int(100*(k+1)/self.ids_str.shape[0]))+"%")
            for l in range(self.ids_str.shape[1]):  
                #print(str('[')+str(k+1)+","+str(l+1)+"]")
                if self.ids_str[k,l] == 'nan':
                    tr_ts_arr[k,l]=np.nan
                else:
                    flag=tr_ts_df[tr_ts_df['ID']==self.ids_str[k,l]]['Flag'].values[0] # filter df by ID and get the flag
                    tr_ts_arr[k,l]=flag
        self.tr_ts_arr = tr_ts_arr
        
    def plots_figs(self):
        # plot
        fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
        title="Train and test pixels"
        ax1.set_title(title)
        ax1.imshow(self.tr_ts_arr)
        plt.axis("off")
        plt.tight_layout()
                
        
        aa=self.labels_[self.tr_ts_arr==1]
        print(np.unique(aa))
        # plot
        fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
        title="All classes"
        ax1.set_title(title)
        ax1.imshow(self.labels)
        plt.axis("off")
        plt.tight_layout()
        
        train_Labels1=np.where(self.tr_ts_arr==0,self.labels,np.nan)
        # plot training parcels
        fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
        title="Train Parcels"
        ax1.set_title(title)
        ax1.imshow(train_Labels1)
        plt.axis("off")
        plt.tight_layout()
        
        train_Labels2=np.where(self.tr_ts_arr==1,self.labels,np.nan)
        # plot training parcels
        fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
        title="Validation parcels"
        ax1.set_title(title)
        ax1.imshow(train_Labels2)
        plt.axis("off")
        plt.tight_layout()
        
        test_Labels=np.where(self.tr_ts_arr==2,self.labels,np.nan)
        # plot
        fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
        title="Test Parcels"
        ax1.set_title(title)
        ax1.imshow(test_Labels)
        plt.axis("off")
        plt.tight_layout()
    
    def plot_map_with_legend(self,lab,values,cmap,crop_typ):       
        plt.figure(figsize=(12,9))
        im = plt.imshow(lab, cmap=cmap,interpolation='none')
        # get the colors of the values, according to the 
        # colormap used by imshow
        colors = [ im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], label=crop_typ[i] ) for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(0.85, 1.01), loc=2, borderaxespad=0.)
        plt.grid(True)
        plt.show()
        plt.axis('off')
        plt.tight_layout()
            
    def Plot_map_with_legend(self):    
           
        self.num_classes=len(self.○)
        values = np.arange(0,self.num_classes,1)    
        cmap = 'jet'
        split_ground_truth.plot_map_with_legend(self.labels_,values,cmap,self.crop_types)
        
        train_Labels1=np.where(self.tr_ts_arr==0,self.labels_,np.nan)
        split_ground_truth.plot_map_with_legend(train_Labels1,values,cmap,self.crop_types)
        
        train_Labels2=np.where(self.tr_ts_arr==1,self.labels_,np.nan)
        split_ground_truth.plot_map_with_legend(train_Labels2,values,cmap,self.crop_types)
        
        test_Labels=np.where(self.tr_ts_arr==2,self.labels_,np.nan)
        split_ground_truth.plot_map_with_legend(test_Labels,values,cmap,self.crop_types)
    
    
    def save_arrays(self,
                    path_flags = "D:\Paper2\Ascending_only_V1\\np_arrays\\V3\\Train_test_flags_asc_Valid",
                    path_Labels_corrected = "D:\Paper2\Ascending_only_V1\\np_arrays\\V3\\Labels_corrected"):
        np.save(path_flags, self.tr_ts_arr, allow_pickle=True, fix_imports=True)
        np.save(path_Labels_corrected, self.labels_, allow_pickle=True, fix_imports=True)
            

#ground_truth=split_ground_truth(split_train=0.6,split_validation=0.2)
#ground_truth.load_arrays()
#ground_truth.call()
#ground_truth.plots_figs()
#ground_truth.Plot_map_with_legend
#ground_truth.save_arrays()
#
#print(ground_truth.tr_ts_df)

