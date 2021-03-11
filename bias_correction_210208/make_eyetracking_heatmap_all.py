# tested in py37

from itertools import product
import pickle
import os
import bdpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import seaborn as sns
import pandas 
import pdb
import math
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
from bdpy.util import makedir_ifnot
from bdpy.ml import cvindex_groupwise
from bdpy.preproc import select_top, average_sample, reduce_outlier, regressout, shift_sample
from bdpy.dataform import append_dataframe
from sklearn.metrics import mean_squared_error
from scipy import ndimage

# data path
im_path = '/home/yqsong/Documents/eye_movement/recon/original_test/'
#results_dir = '/home/kiss/data/fmri_shared/eyetracker/YS210201/eyetracking'

eyetracker_data = [
#     {'file': 'YS210108_ses02_run04.pkl',
#                    'label':'freeviewing_heatmap'},
                   {'file': ['YS210108_ses02_run03.pkl'],
                    'results_dir':'/home/kiss/data/fmri_shared/eyetracker/YS210108/eyetracking',
                    'correction_file':['YS210108_ses02_run01.pkl','YS210108_ses02_run02.pkl'],
                    'linreg_file':'../GOD_eyetracking_210108/bias_correction/',
                    'label':'fixation_heatmap_linreg'},
#                     'label':'fixation_heatmap_no_correction'},
#                    'label':'fixation_heatmap_ms'} ,  
                   {'file': ['YS210128_ses01_run03.pkl'],
                    'results_dir':'/home/kiss/data/fmri_shared/eyetracker/YS210128',
                    'correction_file':['YS210128_ses01_run01.pkl','YS210128_ses01_run02.pkl'],
                    'linreg_file':'../GOD_eyetracking_210128/bias_correction/',
                    'label':'fixation_heatmap_linreg'},
#                     'label':'fixation_heatmap_no_correction'},
#                    'label':'fixation_heatmap_ms'} ,  
                   {'file': ['YS210201_ses01_run05.pkl','YS210201_ses01_run07.pkl'],
                    'results_dir':'/home/kiss/data/fmri_shared/eyetracker/YS210201/eyetracking',
                    'correction_file':['YS210201_ses01_run03.pkl','YS210201_ses01_run04.pkl'],
                    'linreg_file':'../GOD_eyetracking_210201/bias_correction/',
                    'label':'fixation_heatmap_linreg'},
#                     'label':'fixation_heatmap_no_correction'},
#                    'label':'fixation_heatmap_ms'} ,  
]
screen_size = [433., 325.]
screen_res = [1024, 768]
vd = 960 + 136
fixation_x1 = 6
fixation_x2 = -6
fixation_y1 = 6
fixation_y2 = -6

image_label_list = ['n01443537_22563',
                    'n01621127_19020',
                    'n01677366_18182',
                    'n01846331_17038',
                    'n01858441_11077',
                    'n01943899_24131',
                    'n01976957_13223',
                    'n02071294_46212',
                    'n02128385_20264',
                    'n02139199_10398',
                    'n02190790_15121',
                    'n02274259_24319',
                    'n02416519_12793',
                    'n02437136_12836',
                    'n02437971_5013',
                    'n02690373_7713',
                    'n02797295_15411',
                    'n02824058_18729',
                    'n02882301_14188',
                    'n02916179_24850',
                    'n02950256_22949',
                    'n02951358_23759',
                    'n03064758_38750',
                    'n03122295_31279',
                    'n03124170_13920',
                    'n03237416_58334',
                    'n03272010_11001',
                    'n03345837_12501',
                    'n03379051_8496',
                    'n03452741_24622',
                    'n03455488_28622',
                    'n03482252_22530',
                    'n03495258_9895',
                    'n03584254_5040',
                    'n03626115_19498',
                    'n03710193_22225',
                    'n03716966_28524',
                    'n03761084_43533',
                    'n03767745_109',
                    'n03941684_21672',
                    'n03954393_10038',
                    'n04210120_9062',
                    'n04252077_10859',
                    'n04254777_16338',
                    'n04297750_25624',
                    'n04387400_16693',
                    'n04507155_21299',
                    'n04533802_19479',
                    'n04554684_53399',
                    'n04572121_3262']

def preprocess(dat_track):
    dat_track = dat_track.dropna(subset=['time'])    
    dat_track = dat_track[dat_track["trial_type"]>0] # Target labels
    return dat_track

def angle2pixel(fixation_x,fixation_y):
    screen_width_va = 180. * (2 * math.atan(screen_size[0] / (2. * vd))) / math.pi
    screen_height_va = 180. * (2 * math.atan(screen_size[1] / (2. * vd))) / math.pi
    fp_pos_pix_x = screen_res[0] * fixation_x / screen_width_va + screen_res[0] / 2.
    fp_pos_pix_y = screen_res[1] * fixation_y / screen_height_va + screen_res[1] / 2.
    
    return fp_pos_pix_x,fp_pos_pix_y

def image_pixel(fixation_x1,fixation_y1,fixation_x2,fixation_y2):
    x1,y1 = angle2pixel(fixation_x1,fixation_y1) 
    x2,y2 = angle2pixel(fixation_x2,fixation_y2)
    return x1-x2,y1-y2

def GaussianMask(sizex,sizey, sigma=33, center=None,fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    # not used 
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)
    
    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0])==False and np.isnan(center[1])==False:            
            x0 = center[0]
            y0 = center[1]        
        else:
            return np.zeros((sizey,sizex))

    return fix*np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

def Fixpos2Densemap(fix_arr, W, H, imgfile, alpha=0.5, threshold=5):
    """
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : marge rate imgfile and heatmap (optional)
    threshold : heatmap threshold(0~255)
    return heatmap 
    """
    heatmap = fix_arr
#     heatmap = np.zeros((H,W), np.float32)
#     for n_subject in tqdm(range(fix_arr.shape[0])):
#         heatmap += GaussianMask(W, H, 33, (fix_arr[n_subject,0],fix_arr[n_subject,1]),
#                                 fix_arr[n_subject,2])

    # Normalization
#     heatmap = heatmap/np.amax(heatmap)
    heatmap = heatmap*255
    heatmap = heatmap.astype("uint8")
    
    if imgfile.any():
        # Resize heatmap to imgfile shape 
        h, w, _ = imgfile.shape
        heatmap = cv.resize(heatmap, (w, h))
        heatmap_color = cv.applyColorMap(heatmap, cv.COLORMAP_HOT)
#         heatmap_color = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        
        
        # Create mask
        mask = np.where(heatmap<=threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images
#         imgfile = cv.cvtColor(imgfile, cv.COLOR_BGR2RGB)
        marge = imgfile*mask + heatmap_color*(1-mask)
        marge = marge.astype("uint8")
        marge = cv.addWeighted(imgfile, 1-alpha, marge,alpha,0)
#         plt.scatter(500//2, 500//2, marker='+', s=64, c='white')
        cv.drawMarker(marge, (h//2, w//2), (255, 255, 255), markerSize=50, thickness=3)
        return marge

    else:
        heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        return heatmap

def create_heatmap(img, fix_arr):
    H, W, _ = img.shape
    
    fix_arr -= fix_arr.min()
    fix_arr /= fix_arr.max()
    fix_arr[:,0] *= W
    fix_arr[:,1] *= H
    
    # Create heatmap
    heatmap = Fixpos2Densemap(fix_arr, W, H, img, 0.7, 5)
    return heatmap


# main loop

for data_idx,data in enumerate(eyetracker_data):
    dat_free = []
    for ii,track_file in enumerate(data['file']): 
        results_file = os.path.join(data['results_dir'], track_file)
        # load eyetracker data
        with open(results_file, 'rb') as f:
            dat_track = pickle.load(f)
        dat_track = preprocess(dat_track)
        if ii>0:
            dat_free = pd.concat( [dat_free, dat_track], axis=0 )
        else:
            dat_free = dat_track
    print(dat_free.shape)
    
#     correct bias 

    loaded_model_x = pickle.load(open(os.path.join(data['linreg_file'], 'correction_model_x.sav'), 'rb'))
    x_test = dat_free['pos_x']
    x_corrected = loaded_model_x.predict(x_test.values.reshape(-1, 1))
    dat_free['pos_x'] = x_corrected
    loaded_model_y = pickle.load(open(os.path.join(data['linreg_file'], 'correction_model_y.sav'), 'rb'))
    y_test = dat_free['pos_y']
    y_corrected = loaded_model_y.predict(y_test.values.reshape(-1, 1))
    dat_free['pos_y'] = y_corrected
    
#     mean subtraction
#     correction_file = data['correction_file']
#     dat_fix = []
#     for ii,track_file in enumerate(correction_file): 
#         results_file = os.path.join(data['results_dir'], track_file)
#         # load eyetracker data
#         with open(results_file, 'rb') as f:
#             dat_track = pickle.load(f)
#         dat_track = preprocess(dat_track)
#         if ii>0:
#             dat_fix = pd.concat( [dat_fix, dat_track], axis=0 )
#         else:
#             dat_fix = dat_track
#     print(dat_fix.shape)
     
#     x_mean = dat_fix["pos_x"]-dat_fix["fp_pos_pix_x"]
#     y_mean = dat_fix["pos_y"]-dat_fix["fp_pos_pix_y"]
#     # subtract mean value from the data
#     dat_free['pos_x'] -= x_mean.mean()
#     dat_free['pos_y'] -= y_mean.mean()
#     print(x_mean.mean(),y_mean.mean())

    if data_idx>0:
        dat_all = pd.concat( [dat_all, dat_free], axis=0 )
    else:
        dat_all = dat_free        
    dir_label = data['label']

      
    
for i, image_label in enumerate(image_label_list):
#     fig = plt.figure(figsize=(3.5, 3.5), dpi=100)

    data_tmp = dat_all.loc[(dat_all['stimulus_name'] == image_label)]
    original_image= image_label + '.JPEG'    
    original_image_path = os.path.join(im_path, original_image)
    img = cv.imread(original_image_path,cv.IMREAD_UNCHANGED)
    h,w,_ = img.shape

    bin_size = 1
    x = np.arange(0, h // bin_size, 1) + 1 
    y = np.arange(0, w // bin_size, 1) + 1
    screen_w,screen_h = image_pixel(fixation_x1,fixation_y1,fixation_x2,fixation_y2)

    x_mesh, y_mesh = np.meshgrid(x, y)
    eye_pos_dense = np.zeros((h // bin_size, w // bin_size)) # Y axis dir: Up-to-down

    for epx, epy in zip(np.round(data_tmp['pos_x']), np.round(data_tmp['pos_y'])):
        # subtract and get the fixations only within the image area
        epx = epx - (512-screen_h//2)
        epy = epy - (384-screen_w//2)
#             epx = epx - (512-screen_w//2)
#             epy = epy - (384-screen_h//2)
        if not 0 < epx < h: continue
        if not 0 < epy < w: continue
        eye_pos_dense[int(h - epy / bin_size) - 1, int(epx / bin_size) - 1] += 1


#         eye_pos_dense = ndimage.gaussian_filter(eye_pos_dense, sigma=8)
    eye_pos_dense = ndimage.gaussian_filter(eye_pos_dense, sigma=1)

    heatmap = create_heatmap(img, eye_pos_dense)
    output_file = './figures/'+dir_label+'/'+image_label+'.JPEG'
    cv.imwrite(output_file,heatmap)
