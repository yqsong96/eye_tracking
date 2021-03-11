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
    dat_track = dat_track[dat_track["trial_type"] == 1.0]
    return dat_track

def downsample_eyetracker_x(dat_track, step = 120):
    # to get averaged eyetracker data of a certain frequency    
    a = dat_track["time"]
    b = dat_track["pos_x"]
    # c = dat_track["fixation_point_position_x"]
    average_tracking = np.array([])
    label = np.array([])

    for i, _ in enumerate(a[::step]):
        sub_list = b[i*step:] if (i+1)*step > len(a) else b[i*step:(i+1)*step]  # Condition if the len(my_list) % step != 0
        average = sum(sub_list)/float(len(sub_list)) 
        average_tracking = np.concatenate((average_tracking, average), axis=None)
        # label_list = c[i*step:] if (i+1)*step > len(a) else c[i*step:(i+1)*step]
        # average = sum(label_list)/float(len(label_list)) 
        average = 0
        label = np.concatenate((label, average), axis=None)
    
    return average_tracking,label

def downsample_eyetracker_y(dat_track, step = 120):
    # to get averaged eyetracker data of a certain frequency    
    a = dat_track["time"]
    b = dat_track["pos_y"]
    # c = dat_track["fixation_point_position_y"]
    average_tracking = np.array([])
    label = np.array([])

    for i, _ in enumerate(a[::step]):
        sub_list = b[i*step:] if (i+1)*step > len(a) else b[i*step:(i+1)*step]  # Condition if the len(my_list) % step != 0
        average = sum(sub_list)/float(len(sub_list)) 
        average_tracking = np.concatenate((average_tracking, average), axis=None)
        # label_list = c[i*step:] if (i+1)*step > len(a) else c[i*step:(i+1)*step]
        # average = sum(label_list)/float(len(label_list)) 
        average = 0
        label = np.concatenate((label, average), axis=None)
    
    return average_tracking,label

def pixel2angle(x, axis, dist):
    # to transfer eyetracker data to visual angle 
    # resolution:1024,768; screen size:370,280mm
    if axis == "x": 
        distance_x = (x-512)*370/1024 
        ang_x = np.degrees(np.arctan(distance_x/dist))
    else:
        # distance_x = (384-x)*280/768 #wrong! up side down
        distance_x = (x-384)*370/1024 
        ang_x = np.degrees(np.arctan(distance_x/dist))
    return ang_x

def merge(dat_track,dist):
    average_tracking_x,label_x = downsample_eyetracker_x(dat_track)
    average_tracking_y,label_y = downsample_eyetracker_y(dat_track)

    axis = "x"
    angle_x = np.array([pixel2angle(i,axis,dist) for i in average_tracking_x])
    axis = "y"
    angle_y = np.array([pixel2angle(i,axis,dist) for i in average_tracking_y])

    # get rmse for x and y axis
    mse = mean_squared_error(label_x, angle_x) 
    rmse_x = np.sqrt(mse) 
    mse = mean_squared_error(label_y, angle_y) 
    rmse_y = np.sqrt(mse)
    return angle_x,label_x,angle_y,label_y,rmse_x,rmse_y



data_list = [
        {'eyetracker_data_file': 'YS210108_ses02_run03.pkl',
         'label':                'GOD_fixation',
         'eye_dist':              960+136},
        {'eyetracker_data_file': 'YS210108_ses02_run04.pkl',
         'label':                'GOD_freeviewing',
         'eye_dist':              960+136}
        ]
result = 'GOD_eyetracker.pkl'
method = "eyetracker"
dat_prediction = pd.DataFrame([], columns=['subject', 'roi', 'axis', 'num_voxel', 'condition', 'correct', 'predicted']) 


for data in data_list:
    results_dir = '/home/kiss/data/fmri_shared/eyetracker/YS210108/tracking_data'
    # analysis_name = 'eye_movement_train_OpenEye_test_ClosedEye.py'
    results_file = os.path.join(results_dir, data['eyetracker_data_file'])
    
    with open(results_file, 'rb') as f:
        dat_track = pickle.load(f)
    dat_track = preprocess(dat_track)
    
    # runs
    sbj="YS210108"
    condition = data['label']
    angle_x,label_x,angle_y,label_y,rmse_x,rmse_y = merge(dat_track,data['eye_dist'])
    dat_prediction = append_dataframe(
                               dat_prediction,
                               subject=sbj, roi=None, 
                               axis = "x",  
                               correct = label_x, predicted= angle_x,
                               condition = condition,
                               method = method
    #                            classification_acc = classification_acc_,
    #                            classification_acc_tr = np.mean(classification_acc_tr),
                              )
    
    dat_prediction = append_dataframe(
                               dat_prediction,
                               subject=sbj, roi=None, 
                               axis = "y",  
                               correct = label_y, predicted= angle_y,
                               condition = condition,
                               method = method
    #                            classification_acc = classification_acc_,
    #                            classification_acc_tr = np.mean(classification_acc_tr),
                              )
    print(dat_prediction.shape)


# run 2
# sbj="YS210108"
# condition = "eyetracker_run2"
# angle_x,label_x,angle_y,label_y,rmse_x,rmse_y = merge(dat_track2)
# dat_prediction = append_dataframe(
#                            dat_prediction,
#                            subject=sbj, roi=None, 
#                            axis = "x",  
#                            correct = label_x, predicted= angle_x,
#                            RMSE = rmse_x, condition = condition,
# #                            classification_acc = classification_acc_,
# #                            classification_acc_tr = np.mean(classification_acc_tr),
#                            corrcoef=np.corrcoef(label_x, angle_x)[0,1],
#                            corrcoef_cv=None,
#                            corrcoef_cv_tr=None
#                           )

# dat_prediction = append_dataframe(
#                            dat_prediction,
#                            subject=sbj, roi=None, 
#                            axis = "y",  
#                            correct = label_y, predicted= angle_y,
#                            RMSE = rmse_y, condition = condition,
# #                            classification_acc = classification_acc_,
# #                            classification_acc_tr = np.mean(classification_acc_tr),
#                            corrcoef=np.corrcoef(label_y, angle_y)[0,1],
#                            corrcoef_cv=None,
#                            corrcoef_cv_tr=None
#                           )
# print(dat_prediction.shape)






results_dir = '/home/yqsong/Documents/eye_movement/GOD_eyetracking/results'
results_file = os.path.join(results_dir, result)
with open(results_file, 'wb') as f:
    pickle.dump(dat_prediction, f)
print("results saved:",dat_prediction.shape)
