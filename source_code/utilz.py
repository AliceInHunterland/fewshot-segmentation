## On the Texture Bias for Few-Shot CNN Segmentation, Implemented by Reza Azad ##
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import random 
import cv2
import matplotlib.pyplot as plt
import copy
import imgaug as ia
import imgaug.augmenters as iaa
from skimage.transform import resize

## Generate Train and Test classes
def Get_tr_te_lists(opt, t_l_path):
    text_file = open(t_l_path, "r")
    Test_list = [x.strip() for x in text_file] 
    Class_list = os.listdir(opt.data_path)
    Train_list = []
    for idx in range(len(Class_list)):
      Train_list.append(Class_list[idx])
    
    return Train_list, Test_list

def get_corner(X):
    corners = np.array([0, 0, 0, 0])
    corners[1] = X.shape[0]-1
    corners[3] = X.shape[1]-1
    while (np.sum(np.sum(X[corners[0], :, 0])))==0:
          corners[0] += 1
    while (np.sum(np.sum(X[corners[1] , :, 0])))==0:
          corners[1] -= 1
    while (np.sum(np.sum(X[:, corners[2], 0])))==0:
          corners[2] += 1
    while (np.sum(np.sum(X[:, corners[3], 0])))==0:
          corners[3] -= 1                    
    return  corners   

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
               sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.25)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 1.5))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)

## Gen k-shot episode for query and support set
def get_episode(opt, setX):
    indx_c = random.sample(range(0, len(setX)), opt.nway)
    indx_s = random.sample(range(1, opt.class_samples+1), opt.class_samples)


    support = np.zeros([opt.nway, opt.kshot, opt.img_h, opt.img_w, 3], dtype = np.float32)
    smasks  = np.zeros([opt.nway, opt.kshot, 56,        56,        1], dtype = np.float32)
    query   = np.zeros([opt.nway,            opt.img_h, opt.img_w, 3], dtype = np.float32)      
    qmask   = np.zeros([opt.nway,            opt.img_h, opt.img_w, 1], dtype = np.float32)  
    #print(opt.class_samples+1)
    #print(opt.kshot)
    for idx in range(len(indx_c)):
        s = np.zeros([ opt.kshot+1, opt.img_h, opt.img_w, 3], dtype = np.float32)
        m = np.zeros([ opt.kshot+1, opt.img_h, opt.img_w,        1], dtype = np.float32)
        for idy in range(opt.kshot+1): # For support set 
            #print(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.jpg')
            try:
              s_img = cv2.imread("/content/fewshot_data/fewshot_data/" + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.jpg' )
              s_msk = cv2.imread("/content/fewshot_data/fewshot_data/" + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.png' )            
              s_img = cv2.resize(s_img,(opt.img_h, opt.img_w))
              s_msk = cv2.resize(s_msk,(opt.img_h, opt.img_w))       
            except:
              print("/content/fewshot_data/fewshot_data/" + setX[indx_c[idx]] + '/' + str(indx_s[idy]) ) 
            s_msk = s_msk /255.
            s_msk = np.where(s_msk > 0.5, 1., 0.)

            
            s[idy] = s_img
            m[idy]  = s_msk[:,:,0:1]
        #print(support[idx].shape)
        #print(smasks[idx].shape)
        tsupport = s.astype(np.uint8)
        tsmasks = m
        try:
            s, m = seq(images=tsupport,  heatmaps=tsmasks)
        except:
            pass
        query[idx] = s[-1]
        qmask[idx] = m[-1]

        support[idx] = s[0:-1]
        for n,i  in enumerate(m[0:-1]):
          smasks[idx, n] = resize(m[n,:,:,:], smasks.shape[2:], anti_aliasing=True)

        #tqim = query[idx].astype(np.int8)       
        #tqmsk = qmask[idx].astype(np.float32) 
        #print(tqim.shape)
        #print(tqmsk.shape)
        #query[idx], qmask[idx] = seq(images=tqim,  heatmaps=tqmsk)
        

    #support, smasks = seq(images=support, heatmaps=smasks)
    #q_img, qmask = seq(images=q_img, heatmaps=qmask)
    support = support /255.
    query   = query   /255.
   
    return support, smasks, query, qmask

## Gen k-shot episode for query and support set
def get_episode_weakannotation(opt, setX):
    indx_c = random.sample(range(0, len(setX)), opt.nway)
    indx_s = random.sample(range(1, opt.class_samples+1), opt.class_samples)

    support = np.zeros([opt.nway, opt.kshot, opt.img_h, opt.img_w, 3], dtype = np.float32)
    smasks  = np.zeros([opt.nway, opt.kshot, 56,        56,        1], dtype = np.float32)
    query   = np.zeros([opt.nway,            opt.img_h, opt.img_w, 3], dtype = np.float32)      
    qmask   = np.zeros([opt.nway,            opt.img_h, opt.img_w, 1], dtype = np.float32)  
                
    for idx in range(len(indx_c)):
        for idy in range(opt.kshot): # For support set 
            s_img = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.jpg' )
            s_msk = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.png' )
            cc = get_corner(s_msk)
            s_msk[cc[0]:cc[1], cc[2]:cc[3], :] = 255
            s_img = cv2.resize(s_img,(opt.img_h, opt.img_w))
            s_msk = cv2.resize(s_msk,(56,        56))        
            s_msk = s_msk /255.
            s_msk = np.where(s_msk > 0.5, 1., 0.)
            support[idx, idy] = s_img
            smasks[idx, idy]  = s_msk[:, :, 0:1] 
        for idy in range(1): # For query set consider 1 sample per class
            q_img = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy+opt.kshot]) + '.jpg' )
            q_msk = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy+opt.kshot]) + '.png' )
            q_img = cv2.resize(q_img,(opt.img_h, opt.img_w))
            q_msk = cv2.resize(q_msk,(opt.img_h, opt.img_w))        
            q_msk = q_msk /255.
            q_msk = np.where(q_msk > 0.5, 1., 0.)
            query[idx] = q_img
            qmask[idx] = q_msk[:, :, 0:1]        

    support = support /255.
    query   = query   /255.
   
    return support, smasks, query, qmask
        
def compute_miou(Es_mask, qmask):
    ious = 0.0
    Es_mask = np.where(Es_mask> 0.5, 1. , 0.)
    for idx in range(Es_mask.shape[0]):
        notTrue = 1 -  qmask[idx]
        union = np.sum(qmask[idx] + (notTrue * Es_mask[idx]))
        intersection = np.sum(qmask[idx] * Es_mask[idx])
        ious += (intersection / union)
    miou = (ious / Es_mask.shape[0])
    return miou
    
    