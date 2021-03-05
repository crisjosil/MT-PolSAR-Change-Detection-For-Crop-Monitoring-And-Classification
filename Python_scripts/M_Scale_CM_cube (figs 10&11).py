# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:11:30 2021

@author: crs2
"""
########################################################## Imports #################################
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import matplotlib.patches as mpatches
from pickle import dump
from pickle import load
########################################################## functions #################################
def plot_map_with_legend(lab,values,cmap,crop_typ):       
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
    
def NN_predict_cube(cube_to_predict_1d,model,labels,title):   
    y_pred_cube_red = np.argmax(model.predict(cube_to_predict_1d), axis=-1)
    y_pred_2d_red = y_pred_cube_red.reshape(labels.shape[0],labels.shape[1])
    y_pred_2d_red=np.where(np.isnan(labels),np.nan,y_pred_2d_red)
    
    fig,(ax1, ax2)=plt.subplots(nrows=1, ncols=2,sharex=True,sharey=True,figsize=(19,9))
    fig.suptitle(title)
    ax1.imshow(labels)
    ax2.imshow(y_pred_2d_red)
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    
root = "D:\\Paper2\\Ascending_only_V1\\np_arrays\\Change_matrix\\V1\\"
layer_name = "CM_Datacube_viz_abs_with_diagV6.npy"
cube_to_predict = np.load(root + layer_name,  allow_pickle=True) 

# load the scaler
layer_name = 'scaler.pkl'
sc = load(open('D:\\Paper2\\V7 final\\' + layer_name, 'rb'))
# transform the test dataset
cube_to_predict_scaled = sc.transform(cube_to_predict)

np.save('D:\\Paper2\\V7 final\\cube_to_predict_scaled', cube_to_predict_scaled, allow_pickle=True, fix_imports=True)

# Upload the scaled cube to drive (7 Gb),
# then use colab to load the model and predict on the scaled cube.

# This could have been done in colab directly but the RAM crashed every time I tried doing sc.transform (7 Gb)
# Load the model could be done here if the same tf and keras version used to train the model were installed. 
# They are not in the office pc and is nightmare to install new things.











############################################################################
# training labels to create the custom loss funciton and then be able to load the saved model
root = "D:\\Paper2\\Ascending_only_V1\\np_arrays\\Final_datasets\\CM\\V7\\"
layer_name = "y_train_with_diag.npy"
y_train_CM0=np.load(root + layer_name,  allow_pickle=True)

# Determine weight of each class according to number of samples
# For train
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train_CM0), y_train_CM0)
class_weight_dict = dict(enumerate(class_weights/class_weights.max())) # normalize the weights to one
sample_weight=np.zeros(y_train_CM0.shape)
for r in np.unique(y_train_CM0):
  sample_weight=np.where(y_train_CM0==r,class_weight_dict[r],sample_weight)

# Array with weights
w_array = np.ones((10,10))
for i in range(10):
  w_array[:,i]=class_weight_dict[i]
for i in range(10):
  w_array[i,i]=1

# name the loss function
ncce = partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'

#Load model
save_model_name = "D:\Paper2\V7 final/CM_diag_V7_Nadam.h5"
model_CS = load_model(save_model_name+'.h5', custom_objects={'w_categorical_crossentropy': ncce})

# labels
root="/content/drive/My Drive/Datasets/AgriSAR_Paper2/Datacube/"
layer_name = "Labels_corrected.npy"#"Areas_asc_FQ2.npy"
print("loading "+layer_name)
Labels=np.load(root+layer_name,  allow_pickle=True)
crop_types = ['Barley','Canary Seed','Canola','Durum wheat','Field Pea','Flax','Lentil','Mixed pasture','Oat','Spring wheat']
values = np.arange(0,9,1) 
#cmap = 'jet'
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["pink","green","blue","red","darkviolet","yellow","k","darkorange","lime",'brown'])
plot_map_with_legend(Labels,values,cmap,crop_types)

print("Reshaping as 2d images ...")
cube_to_predict_scaled=cube_to_predict_scaled.reshape(cube_to_predict_scaled.shape[0],7,7,3)
NN_predict_cube(cube_to_predict,model_CS,Labels,"")