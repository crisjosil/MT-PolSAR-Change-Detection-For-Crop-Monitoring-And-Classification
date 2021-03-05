# MT-PolSAR-Change-Detection-For-Crop-Monitoring-And-Classification
Repository associated with the paper: Multi-Temporal Polarimetric SAR (MT-PolSAR) Change Detection For Crop Monitoring And Crop Type Classification

A detailed description to reproduce the work presented in the paper can be found [here](https://github.com/crisjosil/MT-PolSAR-Change-Detection-For-Crop-Monitoring-And-Classification/blob/master/Python_scripts/Steps_to_reproduce_short.txt). In summary you can reproduce the results of the paper running the scripts in the following order:

1. Python_scripts/B1_array2D_of_coherency_matrices.py, 
2. Python_scripts/E_MT_Datacube.py,
3. Python_scripts/H2a_Change_mat_datacube_other_viz_abs.py,
4. Python_scripts/I4_master.py, 
5. Colab_1: AgriSAR_train_NNs.ipynb,
6. Colab_2: AgriSAR_test_NNs.ipynb,
7. Python_scripts/M_Scale_CM_cube (figs 10&11).py,
8. Colab_3: AgriSAR_Prediction_maps.ipynb. 

## Paper Abstract:
The interpretation of multidimensional Synthetic
Aperture Radar (SAR) data often requires expert knowledge for
simultaneous consideration of several time series of polarimetric
features to visualise and understand the physical changes of a
target and the temporal evolution between the SAR signal and
a target on the ground. Multitemporal polarimetric SAR (MTPolSAR) change detection has been introduced in the literature
in [1] and [2] in an effort to solve this by characterising the
changes over time. However, the obtained results either only
exploit intensity of changes or the resulting changed scattering
mechanisms are not guaranteed to represent physical changes of
the target.
This paper presents a variation in the change detector used
in [2] based on the difference of covariance matrices that
characterise the polarimetric information of a resolution cell,
allowing for an intuitive representation and characterisation of
physical changes of a target and its dynamics. We show the
results of this method for monitoring growth stages of rice crops
and we present a novel application of the method in which the
capabilities for image classification are investigated applying it
to crop type mapping from MT-PolSAR data. We compare its
performance with a neural network-based classifier that uses time
series of PolSAR features derived from target covariance matrix
decomposition as input.
Experimental results show that the classification performance
of the method presented here and the baseline are comparable,
with differences between the two methods in the overall balanced
accuracy and the F1-macro metrics of around 2% and 3%,
respectively. The method presented here achieves similar classification performances than a traditional PolSAR data classifier
while providing additional advantages in terms of interpretability
and insights about the physical changes of a target over time.

## Main results
### Rice multitemporal polarimetric change over time
Typical rice change matrix. Left: Change matrix. Top right: Main rice growth stages. Bottom right: RGB
interpretation of added and removed scattering mechanisms. The added and removed SMs between two stages correspond to
their intersecting squares in the upper and lower triangular part, respectively
![Rice_change_matrix](https://user-images.githubusercontent.com/38487043/110109378-42e03d80-7da5-11eb-9217-29c0488133c5.png)

### Canola multitemporal polarimetric change over time
<img src=https://user-images.githubusercontent.com/38487043/110109518-7b801700-7da5-11eb-9bdf-b637ce7ff9e2.png" width="200" height="200">

### Prediction maps with 10 crop types
![Prediction_map](https://user-images.githubusercontent.com/38487043/110109596-95b9f500-7da5-11eb-9658-cd657c47f641.PNG)

### Prediction maps with 6 crop types
![Prediction_map_red](https://user-images.githubusercontent.com/38487043/110109618-9eaac680-7da5-11eb-87e2-adec39ad18c1.PNG)
 


