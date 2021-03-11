## env: py37

from itertools import product
import pickle
import os
import bdpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import seaborn as sns
import pandas 
import pdb
import numpy as np
import pandas as pd
from bdpy.util import makedir_ifnot
from bdpy.ml import cvindex_groupwise
from bdpy.preproc import select_top, average_sample, reduce_outlier, regressout, shift_sample
from bdpy.dataform import append_dataframe
from sklearn.metrics import mean_squared_error


def preprocess(dat_track):
    dat_track = dat_track.dropna(subset=['time'])
    dat_track = dat_track[dat_track["trial_type"] == 5.0]
    return dat_track

def downsample_eyetracker_x(dat_track, step = 120):
    # to get averaged eyetracker data of a certain frequency    
    a = dat_track["time"]
    b = dat_track["pos_x"]
    c = dat_track["fixation_point_position_x"]
    average_tracking = np.array([])
    label = np.array([])

    for i, _ in enumerate(a[::step]):
        sub_list = b[i*step:] if (i+1)*step > len(a) else b[i*step:(i+1)*step]  # Condition if the len(my_list) % step != 0
        average = sum(sub_list)/float(len(sub_list)) 
        average_tracking = np.concatenate((average_tracking, average), axis=None)
        label_list = c[i*step:] if (i+1)*step > len(a) else c[i*step:(i+1)*step]
        average = sum(label_list)/float(len(label_list)) 
        label = np.concatenate((label, average), axis=None)
    
    return average_tracking,label

def downsample_eyetracker_y(dat_track, step = 120):
    # to get averaged eyetracker data of a certain frequency    
    a = dat_track["time"]
    b = dat_track["pos_y"]
    c = dat_track["fixation_point_position_y"]
    average_tracking = np.array([])
    label = np.array([])

    for i, _ in enumerate(a[::step]):
        sub_list = b[i*step:] if (i+1)*step > len(a) else b[i*step:(i+1)*step]  # Condition if the len(my_list) % step != 0
        average = sum(sub_list)/float(len(sub_list)) 
        average_tracking = np.concatenate((average_tracking, average), axis=None)
        label_list = c[i*step:] if (i+1)*step > len(a) else c[i*step:(i+1)*step]
        average = sum(label_list)/float(len(label_list)) 
        label = np.concatenate((label, average), axis=None)
    
    return average_tracking,label

def pixel2angle(x, axis):
    # to transfer eyetracker data to visual angle 
    # resolution:1024,768; screen size:370,280mm
    if axis == "x": 
        distance_x = (x-512)*370/1024 
        ang_x = np.degrees(np.arctan(distance_x/1096))
    else:
        # distance_x = (384-x)*280/768 #wrong! up side down
        distance_x = (x-384)*280/768   
        ang_x = np.degrees(np.arctan(distance_x/1096))
    return ang_x

def merge(dat_track):
    average_tracking_x,label_x = downsample_eyetracker_x(dat_track)
    average_tracking_y,label_y = downsample_eyetracker_y(dat_track)

    axis = "x"
    angle_x = np.array([pixel2angle(i,axis) for i in average_tracking_x])
    axis = "y"
    angle_y = np.array([pixel2angle(i,axis) for i in average_tracking_y])

    # get rmse for x and y axis
    mse = mean_squared_error(label_x, angle_x) 
    rmse_x = np.sqrt(mse) 
    mse = mean_squared_error(label_y, angle_y) 
    rmse_y = np.sqrt(mse)
    return angle_x,label_x,angle_y,label_y,rmse_x,rmse_y

results_dir = '/home/kiss/data/fmri_shared/eyetracker/YS210108/tracking_data'
# analysis_name = 'eye_movement_train_OpenEye_test_ClosedEye.py'
results_file = os.path.join(results_dir, 'YS210108_ses02_run01.pkl')

with open(results_file, 'rb') as f:
    dat_track1 = pickle.load(f)
dat_track1 = preprocess(dat_track1)
    
results_dir = '/home/kiss/data/fmri_shared/eyetracker/YS210108/tracking_data'
# analysis_name = 'eye_movement_train_OpenEye_test_ClosedEye.py'
results_file = os.path.join(results_dir, 'YS210108_ses02_run02.pkl')

with open(results_file, 'rb') as f:
    dat_track2 = pickle.load(f)
# print (dat_track2.shape)
dat_track2 = preprocess(dat_track2)

#dat_prediction = pd.DataFrame([], columns=['subject', 'roi', 'axis', 'num_voxel', 'condition', 'correct', 'predicted', 'RMSE','corrcoef','corrcoef_cv', 'corrcoef_cv_tr']) 
dat_prediction = pd.DataFrame([], columns=['subject', 'roi', 'axis', 'num_voxel', 'condition', 'correct', 'predicted']) 

# run 1
sbj="YS210108"
condition = "eyetracker_run1"

# pdb.set_trace()
angle_x,label_x,angle_y,label_y,rmse_x,rmse_y = merge(dat_track1)
dat_prediction = append_dataframe(
                           dat_prediction,
                           subject=sbj, roi=None, 
                           axis = "x",  
                           correct = label_x, predicted= angle_x,
                           RMSE = rmse_x, condition = condition,
#                            classification_acc = classification_acc_,
#                            classification_acc_tr = np.mean(classification_acc_tr),
                           # corrcoef=np.corrcoef(label_x, angle_x)[0,1],
                           # corrcoef_cv=None,
                           # corrcoef_cv_tr=None
                          )

dat_prediction = append_dataframe(
                           dat_prediction,
                           subject=sbj, roi=None, 
                           axis = "y",  
                           correct = label_y, predicted= angle_y,
                           RMSE = rmse_y, condition = condition,
#                            classification_acc = classification_acc_,
#                            classification_acc_tr = np.mean(classification_acc_tr),
                           # corrcoef=np.corrcoef(label_y, angle_y)[0,1],
                           # corrcoef_cv=None,
                           # corrcoef_cv_tr=None
                          )
print(dat_prediction.shape)


# run 2
sbj="YS210108"
condition = "eyetracker_run2"
angle_x,label_x,angle_y,label_y,rmse_x,rmse_y = merge(dat_track2)
dat_prediction = append_dataframe(
                           dat_prediction,
                           subject=sbj, roi=None, 
                           axis = "x",  
                           correct = label_x, predicted= angle_x,
                           RMSE = rmse_x, condition = condition,
#                            classification_acc = classification_acc_,
#                            classification_acc_tr = np.mean(classification_acc_tr),
                           # corrcoef=np.corrcoef(label_x, angle_x)[0,1],
                           # corrcoef_cv=None,
                           # corrcoef_cv_tr=None
                          )

dat_prediction = append_dataframe(
                           dat_prediction,
                           subject=sbj, roi=None, 
                           axis = "y",  
                           correct = label_y, predicted= angle_y,
                           RMSE = rmse_y, condition = condition,
#                            classification_acc = classification_acc_,
#                            classification_acc_tr = np.mean(classification_acc_tr),
                           # corrcoef=np.corrcoef(label_y, angle_y)[0,1],
                           # corrcoef_cv=None,
                           # corrcoef_cv_tr=None
                          )
print(dat_prediction.shape)







results_dir = '/home/yqsong/Documents/eye_movement/GOD_eyetracking/results'
results_file = os.path.join(results_dir, "eye_movement_eyetracker.pkl")
with open(results_file, 'wb') as f:
    pickle.dump(dat_prediction, f)
print("results saved:",dat_prediction.shape)