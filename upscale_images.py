"""
This is used to enhance images using EDSR super resolution

Download and store the EDSR_x3.pb files from github and change the 'pbPath' variable to point to the '.pb' files
"""

import tensorflow as tf
import tensorflow.compat.v1 as tf
import os
import cv2
import numpy as np
import math
#import data_utils
from skimage import io
#import edsr
from PIL import Image

#from tensorflow.python.tools import freeze_graph
#from tensorflow.python.tools import optimize_for_inference_lib
#from tensorflow.tools.graph_transforms import TransformGraph




def upscaleFromPb():
    """
    Upscale single image by desired model. This loads a .pb file.
    """
    # Read model
    pbPath = "/home/ld07063u/sr_models/EDSR_x3.pb"

    # Get graph
    graph = load_pb(pbPath)

    mean = [103.1545782, 111.561547, 114.35629928]

    fullimg = cv2.imread('/home/ld07063u/generated_images_64_batch_1200_5000_imgs/gen_image_48.jpg', 3)
    floatimg = fullimg.astype(np.float32) - mean
    LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

    LR_tensor = graph.get_tensor_by_name("IteratorGetNext:0")
    HR_tensor = graph.get_tensor_by_name("NHWC_output:0")

    with tf.Session(graph=graph) as sess:
        print("Loading pb...")
        output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})
        Y = output[0]
        HR_image = (Y + mean).clip(min=0, max=255)
        HR_image = (HR_image).astype(np.uint8)

        #bicubic_image = cv2.resize(fullimg, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        #cv2.imshow('Original image', fullimg)
        print('writing gile')
        cv2.imwrite('/home/ld07063u/generated_images_64_batch_1200_5000_imgs/gen_image_48_x3.jpg', HR_image)
        #cv2.imshow('Bicubic upscaled image', bicubic_image)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    sess.close()

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

#upscaleFromPb()


pbPath = "/home/ld07063u/sr_models/EDSR_x4.pb"

    # Get graph
graph = load_pb(pbPath)

mean = [103.1545782, 111.561547, 114.35629928]

#img_list = [18,31,289,306,352,369,388,546,581,771,1165,1170,1217,1362,1424,1932,2084,2218,2289,2312,2378,2410,2476,2656,2774,2786,2846,2848,2876,3012,3026,3079,3093,3144,3332,4165,4182,4600,4651,4667,4687,4755,4769,4873,4973]

img_list = [302,2,102,148]#48,328,339,761,813,867,914,1122,2300,2667,2765,3361,3763,4122,4251,4301,4670,4992]

with tf.Session(graph=graph) as sess:
        print("Loading pb...")
        for k,i in enumerate(img_list):
    
            print('processing ',k,' / ',len(img_list))
            fullimg = cv2.imread('/home/ld07063u/ls_gan_edsr_sample/gen_image_'+str(i)+'_x4.jpg', 3)
            floatimg = fullimg.astype(np.float32) - mean
            LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

            LR_tensor = graph.get_tensor_by_name("IteratorGetNext:0")
            HR_tensor = graph.get_tensor_by_name("NHWC_output:0")
            
            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})
            Y = output[0]
            HR_image = (Y + mean).clip(min=0, max=255)
            HR_image = (HR_image).astype(np.uint8)
            
            print('writing file ',i)
            cv2.imwrite('/home/ld07063u/ls_gan_edsr_sample/gen_image_'+str(i)+'_x4x4.jpg', HR_image)
            #cv2.imshow('Bicubic upscaled image', bicubic_image)
sess.close()
print('finished processing')