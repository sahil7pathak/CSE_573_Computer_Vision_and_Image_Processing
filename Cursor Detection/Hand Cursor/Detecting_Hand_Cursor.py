'''
Title          :- Project 1, Task 3 - Cursor Detection
Author         :- Sahil Suhas Pathak
UBITName       :- sahilsuh
Person Number  :- 50289739

'''
#Detecting hand cursor 
#Successfully detected - 't1_2.jpg','t1_3.jpg','t1_4.jpg' and 't1_6.jpg'
import cv2
from math import exp
from scipy.ndimage.filters import gaussian_filter
import numpy as np
images = ['t1_1.jpg','t1_2.jpg','t1_3.jpg','t1_4.jpg','t1_5.jpg','t1_6.jpg']
for y in images:
    img = cv2.imread(y,0)
    temp = cv2.imread('template_1_hand.png',0)
    temp = temp[::4, ::4] #Resizing required
    w, h = temp.shape[::-1] 
    blur = gaussian_filter(img,0)
    lap_i = cv2.Laplacian(blur, cv2.CV_32F)
    lap_t = cv2.Laplacian(temp, cv2.CV_32F)
    res = cv2.matchTemplate(lap_i, lap_t, cv2.TM_CCORR_NORMED)
    threshold = 0.45 
    loc = np.where( res >= threshold) 
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2) 
    cv2.imshow(y,img)
    cv2.waitKey(0)