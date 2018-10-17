'''
Title          :- Project 1, Task 3 - Cursor Detection
Author         :- Sahil Suhas Pathak
UBITName       :- sahilsuh
Person Number  :- 50289739

'''
#Detecting Normal White Cursor
import cv2
from math import exp
from scipy.ndimage.filters import gaussian_filter
import numpy as np
images = [ 'pos_15.jpg'] #You may pass as many images as you can here in this list
for y in images:
    img = cv2.imread(y,0)
    temp = cv2.imread('template_1_white.png',0)
    
    w, h = temp.shape[::-1] 
    blur = gaussian_filter(img,0)
    lap_i = cv2.Laplacian(blur, cv2.CV_32F)
    lap_t = cv2.Laplacian(temp, cv2.CV_32F)
    res = cv2.matchTemplate(lap_i, lap_t, cv2.TM_CCORR_NORMED)
    threshold = 0.5
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2) 
    cv2.imshow(y,img)
    cv2.waitKey(0)