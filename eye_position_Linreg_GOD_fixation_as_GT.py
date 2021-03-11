import os
import pickle
import pdb
import bdpy
from bdpy.ml import cvindex_groupwise
from bdpy.preproc import select_top, average_sample, reduce_outlier, regressout, shift_sample
from bdpy.dataform import append_dataframe
from bdpy.fig import makefigure, box_off, draw_footnote
from bdpy.util import makedir_ifnot


from scipy.spatial.distance import cdist
from sklearn import linear_model
# from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn import svm 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product


data_dir = '/home/share/data/fmri_shared/eyetracker/YS210108/bdata'
os.listdir(data_dir)
results_dir = '/home/yqsong/Documents/eye_movement/GOD_eyetracking/results'


# Setups

analysis_basename = 'eye_position_Linreg_GOD.py'
makedir_ifnot(os.path.join(results_dir, analysis_basename))

print('----------------------------------------')
print('Loading data')

sbj = 'YS210108'
method = "MRI-based"

cond_type = 5
dir_type = 0

dat_name = 'YS210108_GODeyetracking_eyetracking_volume_native_prep.h5'
# data_all = {}
# for sbj in subjects:
#     data_all[sbj] = bdpy.BData(os.path.join(data_dir, subjects[sbj][0]))
data_train = bdpy.BData(os.path.join(data_dir, dat_name))
dat_name = 'YS210108_GODeyetracking_GOD_volume_native_raw.h5'
data_test = bdpy.BData(os.path.join(data_dir, dat_name))


results = pd.DataFrame([], columns=['subject', 'roi', 'axis', 'num_voxel', 'condition', 
                                    'correct', 'predicted', 'method','train_label']) 

training_data_list = [{'label':'train_run1'},
                      {'label':'train_run2'},
                      {'label':'train_run1&2'},
                      ]
conditions = ['GOD_fixation','GOD_freeviewing']

roi = 'PureEyeBall'
#     'EyeBall'  : 'eyetracker_r*_eyeball = 1',


# num_voxel_list = [2, 5, 10, 20, 30, 50, 75, 100, 200, 300, 400, 500, 1000, 2000, 10000]
# num_voxel_list = [2, 5, 10, 20, 50, 100, 200, 300, 500]
num_voxel = 100

# define functions


def make_group(xpos, ypos):
    pos = np.array(zip(xpos, ypos))
    unique_position_set = np.array(list(set(zip(xpos, ypos))))
    group = np.ones(pos.shape[0]) * -1
    for i in range(unique_position_set.shape[0]):
        counter = 1
        for pi in range(len(pos)):
            if (pos[pi] == unique_position_set[i]).all():
                group[pi] = counter
                #group[pi + 1] = counter
                counter += 1
    return group


def voxelselection_corr(xtrain, ytrain, xtest, ytest, n_voxel=100):
    '''Voxel selection based on f values of correlation abs'''
     
    if ytrain.ndim == 1:
        _ytrain = ytrain.reshape((1, ytrain.shape[0]))
    else:
        _ytrain = ytrain
    corr = 1 - cdist(xtrain.transpose(), _ytrain, metric="correlation")
    ind = np.argsort(np.abs(corr).flatten())[::-1]
    xtrain = xtrain[:, ind[:n_voxel]]
    xtest = xtest[:, ind[:n_voxel]]

    return xtrain, ytrain, xtest, ytest

def select_label(dat, trial_type, circular_type):
    x = dat.select('eyetracker_*_eyeball - pickatlas* - hcp180*')
    y = dat.select('Label')    # Target labels
    trial_t = y[:, 0].flatten()
    xpos = y[:, 1]
    ypos = y[:, 2]
    circular = y[:, 3].flatten()
    eye_closed = y[:, 4].flatten()
    dense_fp = y[:, 5].flatten()
    # -1(init scan), -2(post scan)
    selector = np.logical_and(trial_t == trial_type, circular == circular_type) 
    x = x[selector, :]
    xpos = xpos[selector]
    ypos = ypos[selector]
    return x, xpos, ypos

def select_training_data(data, training_data_label):
   # selector for training:
    x_long, xpos_long, ypos_long = select_label(data, cond_type, dir_type)
    
    selector_run1 = x_long.shape[0] //2
    x_1 = x_long[0:selector_run1, :]
    xpos_1 = xpos_long[0:selector_run1]
    ypos_1 = ypos_long[0:selector_run1]

    x_2 = x_long[selector_run1:, :]
    xpos_2 = xpos_long[selector_run1:]
    ypos_2 = ypos_long[selector_run1:]
    if training_data_label == 'train_run1&2': 
        return x_long, xpos_long, ypos_long
    elif training_data_label == 'train_run1':
        return x_1, xpos_1, ypos_1
    else:
        return x_2, xpos_2, ypos_2
    
def select_test_data(predict,true, condition):
   # selector for training:
    # x_long, xpos_long, ypos_long = select_label(data, cond_type, dir_type)
    
    selector_run1 = len(predict) //2
    x_1 = predict[0:selector_run1]
    y_1 = true[0:selector_run1]
    x_2 = predict[selector_run1:]
    y_2 = true[selector_run1:]

    if condition == 'GOD_fixation': 
        return x_1,y_1
    elif condition == 'GOD_freeviewing':
        return x_2,y_2



    
#preprocess is done when making bdata

runs = data_test.get('Run').flatten()
data_test = data_test.applyfunc(regressout, where='VoxelData', regressor=np.array([[],[]]),
            group=runs, remove_dc=True, linear_detrend=True)
x = data_test.select('eyetracker_*_eyeball - pickatlas* - hcp180*')
y = data_test.select('Label')    # Target labels
trial_t = y[:, 0].flatten()
    # -1(init scan), -2(post scan)
selector = np.logical_not(np.logical_or(trial_t < 0, trial_t == 2))
# selector for test: 
x_te = x[selector, :]



# main loop
for training_data in training_data_list:
    print('--------------------')
    print('Subject:    %s' % sbj)
    print('ROI:        %s' % roi)
    print('Num voxels: %d' % num_voxel)
#     assert 1 == 0

    # Prepare data
    print('Preparing data')
    training_data_label = training_data['label']
    # get training data:
    x_tr, xpos_tr, ypos_tr = select_training_data(data_train, training_data_label)

    print("x_tr.shape", x_tr.shape)
    print("xpos_tr.shape", xpos_tr.shape)
    print("ypos_tr.shape", ypos_tr.shape)
    print("x_test.shape", x_te.shape)
    # print("xpos_test.shape", xpos_test.shape)
    # print("ypos_test.shape", ypos_test.shape)
    #break
    # cv_classifier, cv_predacc = crossvalidation_cls_svm_multiclass(x, y, runs, n_voxel=num_voxel)

    # crossvalidation_cls_svm_multiclass
    
    for ii in range(0,2):
        if ii == 0:
            pos_tr = xpos_tr
            # pos_test = xpos_test
            print ('fixation x:')
        else:
            pos_tr = ypos_tr
            # pos_test = ypos_test
            print ('fixation y:')
#        pdb.set_trace()
        cv_res_tr = []
        y_pred = []
        y_true = []
        cv_res = []
        cv_res_tr = []
        rmse = []
#         classification_acc = []
#         classification_acc_tr = []

        x_train = []
        x_test = []
        y_train = []
        y_test = []
#         cv_index = cvindex_groupwise(new_blocks)
#         for train_index, test_index in cv_index:
        x_train = x_tr
        y_train = pos_tr
        x_test  = x_te
        # y_test  = pos_test

        x_mean_tr = np.mean(x_train, axis=0)
        x_scale_tr = np.std(x_train, axis=0, ddof=1)

        x_train = (x_train - x_mean_tr) / x_scale_tr
        x_test = (x_test - x_mean_tr) / x_scale_tr 
        
        x_train, y_train, x_test, y_test = voxelselection_corr(x_train, y_train, x_test, y_test, n_voxel=num_voxel)
        # try f-value - selection too? floating ->unique value?

        # Result: x_train.shape ; y_train.shape (depending on the number of blocks)

        model = linear_model.Lasso(alpha=0.1)
#             model.fit(x_train, y_train.astype('int'))
        model.fit(x_train, y_train)
        res = model.predict(x_test)
        res_tr = model.predict(x_train)


        res = res.flatten()
        res_tr = res_tr.flatten()

        # rmse
        # mse = mean_squared_error(y_test, res) 
        # rmse = np.sqrt(mse) 

        # corr            
        # corr = np.corrcoef(y_test, res)[0,1]
        y_pred.extend(list(res))
        y_true.extend(list(y_test))
        # cv_res.append(corr)
        corr_tr = np.corrcoef(y_train, res_tr)[0,1]
        cv_res_tr.append(corr_tr)

        # corrcoef_cv_acc = np.nanmean(cv_res)
        # print('Corrcoef cv: {}'.format(corrcoef_cv_acc))  
 
        # correct variable names: prediction_accuracy -> corrcoef 
        for test_condition in conditions:
            y_pred_,y_true_=select_test_data(y_pred,y_true, test_condition)
            results = append_dataframe(
                               results,
                               subject=sbj, roi=roi, 
                               axis = "x" if ii == 0 else "y",
                               num_voxel = num_voxel, 
                               correct = y_true_, predicted= y_pred_,
                               train_label = training_data_label,
                               method = method,
                               condition = test_condition
    #                            classification_acc = classification_acc_,
    #                            classification_acc_tr = np.mean(classification_acc_tr),
                               # corrcoef=np.corrcoef(y_true, y_pred)[0,1],
                               # corrcoef_cv=corrcoef_cv_acc,
                               # corrcoef_cv_tr=np.mean(cv_res_tr)
                              )

print (results.shape)
results_file = os.path.join(results_dir, analysis_basename, "results_MRI_based_GOD.pkl")
with open(results_file, 'wb') as f:
    pickle.dump(results, f)
    
print("Result saved!")


