import os
import cv2
import numpy as np
import random
#from scipy import ndimage
import imutils

def get_seq_img(seqlen, dtheta, angle_start=0, random_start=True, random_seq=False):

    img = cv2.imread('img.png',0)
    h,w = img.shape[0], img.shape[1]

    if random_seq:
        random_start=True
    
    angle0=angle_start    
    if random_start:
        angle0 = random.uniform(0.0,360.0)
        
    seq=[]    
    for i in range(seqlen):
        if random_seq:
            angle_rot = random.uniform(0.0,360.0)
        else:
            angle_rot =  angle0 + dtheta*i
        seq.append(imutils.rotate(img, angle_rot))
        
    return seq




