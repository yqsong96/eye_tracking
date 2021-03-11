'''Make video of eye tracker data.'''


import os
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def main():
    results_dir = '/home/kiss/data/fmri_shared/eyetracker/YS210108/tracking_data'
    # analysis_name = 'eye_movement_train_OpenEye_test_ClosedEye.py'
    
    data_list = [
        {'eyetracker_data_file': 'YS210108_ses02_run01.pkl',
         'label':                'eye_movement_01'},
        {'eyetracker_data_file': 'YS210108_ses02_run02.pkl',
         'label':                'eye_movement_02'},
        # {'eyetracker_data_file': 'YS210108_ses02_run03.pkl',
        #  'label':                'GOD_fixation'},
        # {'eyetracker_data_file': 'YS210108_ses02_run04.pkl',
        #  'label':                'GOD_freeviewing'}
        ]

    for data_file in data_list:
        results_file = os.path.join(results_dir, data_file['eyetracker_data_file'])
        with open(results_file, 'rb') as f:
            data = pickle.load(f)        

        # output_file = os.path.join('./video/', os.path.splitext(os.path.basename(data_file))[0] + '.mp4')
        output_file = os.path.join('./video/', data_file['label'] + '.avi')
        make_movie(data, output_file)
        print('%s saved.' % output_file)


def make_movie(data, output_file):
    canvas_size = (1024, 768)
    fps = 60
    
    # Marker size
    eye_size = 16
    fp_size = 16

    # Init video writer
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize=canvas_size)
    # Codec 0 -> Raw movie

    for index, row in data.iterrows():
        t = row['time']
        eye_pos = [row['pos_x'], row['pos_y']]  # TODO: check y axis direction
        fp_pos = [row['fp_pos_pix_x'], row['fp_pos_pix_y']]

        if not np.isnan(t):
            if np.isnan(fp_pos[0]): fp_pos[0] = canvas_size[0] / 2.
            if np.isnan(fp_pos[1]): fp_pos[1] = canvas_size[1] / 2.

        # Init image and draw object
        img = Image.new('RGB', canvas_size, (128, 128, 128))
        draw = ImageDraw.Draw(img)

        # Caption
        font = ImageFont.truetype('arial.ttf', 24)
        if row['trigger'] == 1:
            draw.text((10, 10), 'Time: %.2f' % t, (0, 0, 0), font=font)
            draw.text((160, 10), 'Trigger IN' % t, (0, 0, 255), font=font)
        else:
            draw.text((10, 10), 'Time: %.2f' % t, (0, 0, 0), font=font)

        draw.text((10, 36), 'Pupil size: %.2f mm' % row['pupil'], (0, 0, 0), font=font)
        draw.text((10, 62), 'Eye position (pixel): (%.2f, %.2f)' % (eye_pos[0], eye_pos[1]), (0, 0, 0), font=font)
        draw.text((10, 88), 'FP position (pixel): (%.2f, %.2f)' % (fp_pos[0], fp_pos[1]), (0, 0, 0), font=font)
        
        # Fixation point
        draw.ellipse((fp_pos[0] - fp_size / 2, fp_pos[1] - fp_size / 2,
                      fp_pos[0] + fp_size / 2, fp_pos[1] + fp_size / 2),
                     fill=(0, 0, 0), outline=(0, 0, 0))

        # Eye position
        draw.rectangle((eye_pos[0] - eye_size / 2, eye_pos[1] - eye_size / 2,
                        eye_pos[0] + eye_size / 2, eye_pos[1] + eye_size / 2),
                       fill=None, outline=(255, 255, 255))

        video.write(np.array(img))

    cv2.destroyAllWindows()
    video.release()

    
if __name__ == '__main__':
    main()
