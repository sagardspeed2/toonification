"""
Internal code snippets were obtained from https://github.com/SystemErrorWang/White-box-Cartoonization/

For it to work tensorflow version 2.x changes were obtained from https://github.com/steubk/White-box-Cartoonization 
"""
import os
import uuid
import time
import subprocess
import sys

import cv2
import numpy as np

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

import network
import guided_filter

class WB_Cartoonize:
    def __init__(self, weights_dir, gpu):
        # load modal with gpu setting
        if not os.path.exists(weights_dir):
            raise FileNotFoundError("Weights Directory not found, check path")
        self.load_model(weights_dir, gpu)
        print("Weights successfully loaded")
    
    # image resize
    def resize_crop(self, image):
        h, w, c = np.shape(image)
        if min(h, w) > 720:
            if h > w:
                h, w = int(720*h/w), 720
            else:
                h, w = 720, int(720*w/h)
        image = cv2.resize(image, (w, h),
                            interpolation=cv2.INTER_AREA)
        h, w = (h//8)*8, (w//8)*8
        image = image[:h, :w, :]
        return image

    # load tensorflow modal
    def load_model(self, weights_dir, gpu):
        # check for tensorflow v1
        try:
            tf.disable_eager_execution()
        except:
            None
        
        # reset tensorflow memory
        tf.reset_default_graph()

        self.input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_image')
        network_out = network.unet_generator(self.input_photo)

        # cartoonize filter
        self.final_out = guided_filter.guided_filter(self.input_photo, network_out, r=1, eps=5e-3)

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)
        
        # set gpu configuration for tensorflow
        if gpu:
            gpu_options = tf.GPUOptions(allow_growth=True)
            device_count = {'GPU':1}
        else:
            gpu_options = None
            device_count = {'GPU':0}
        
        # tensorflow configure with gpu
        config = tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)
        
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, tf.train.latest_checkpoint(weights_dir))

    # cartoonize image
    def infer(self, image):
        image_src = cv2.imread(image)

        # resize image
        image = self.resize_crop(image_src)
        
        #convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clip_hist_percent = 10

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        '''
        # Calculate new histogram with desired range and show histogram 
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        plt.show()
        '''
        # convert image brightness and histogram
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        
        ## Session Run
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})
        
        ## Post Process
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output

if __name__ == '__main__':
    # gpu - command line
    gpu = len(sys.argv) < 2 or sys.argv[1] != '--cpu'
    # white box cartoonize - load model
    wbc = WB_Cartoonize(os.path.abspath('white_box_cartoonizer/saved_models'), gpu)
    # default image
    img = cv2.imread('white_box_cartoonizer/test.jpg')
    # convert color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cartoonize
    cartoon_image = wbc.infer(img)

    # show image
    import matplotlib.pyplot as plt
    plt.imshow(cartoon_image)
    plt.show()
