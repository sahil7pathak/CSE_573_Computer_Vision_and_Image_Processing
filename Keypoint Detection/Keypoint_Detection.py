'''
Title          :- Project 1, Task 2 - Keypoints Detection
Author         :- Sahil Suhas Pathak
UBITName       :- sahilsuh
Person Number  :- 50289739

'''
import math
import numpy as np
import cv2

img = cv2.imread('task2.jpg',0)

#Zero Padding
h, w = img.shape[:2]
h = h + 6
w = w + 6
pad_img = [[0 for x in range(w)] for y in range(h)]

for i in range(len(img)):
    for j in range(len(img[0])):
        pad_img[i+1][j+1] = img[i][j]
        
pad_img = np.asarray(pad_img)

#Flips the kernel
def flip_operator(kernel):
    kernel_copy = [[0 for x in range(kernel.shape[1])] for y in range(kernel.shape[0])]
    #kernel_copy = kernel.copy()
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            kernel_copy[i][j] = kernel[kernel.shape[0]-i-1][kernel.shape[1]-j-1]
    kernel_copy = np.asarray(kernel_copy)
    return kernel_copy

#Convolution Logic
def conv(image, kernel):
    #Flipping the kernel
    kernel = flip_operator(kernel)
    
    img_height = image.shape[0]
    img_width = image.shape[1]
    
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    
    h = kernel_height//2
    w = kernel_width//2
    
    conv_result = [[0 for x in range(img_width)] for y in range(img_height)] 
      
    for i in range(h, img_height-h):
        for j in range(w, img_width-w):
            sum = 0 
            for m in range(kernel_height):
                for n in range(kernel_width):
                    sum = (sum + kernel[m][n]*image[i-h+m][j-w+n])
                    
            conv_result[i][j] = sum
    conv_result = np.asarray(conv_result)   
    return conv_result

#Defines the Gaussian Kernel                     
def gau_kernel(sigma):
    w, h = 7, 7;
    gau_mat = [[0 for x in range(w)] for y in range(h)] 
    for i in range(0,7):
        for j in range(0,7):
            gau_mat[i][j]=(1/(2*math.pi*sigma*sigma))*(math.exp(-(((j-3)**2 + (3-i)**2)/(2*sigma*sigma))))
    gau_mat = np.asarray(gau_mat)
    return gau_mat

#Building octave for every level
def calculate_octave(img, sigma):
    g1 = gau_kernel(sigma[0])
    ga = conv(img,g1)
    ga = np.asarray(ga)
    
    g2 = gau_kernel(sigma[1])
    gb = conv(img,g2)
    gb = np.asarray(gb)
    
    g3 = gau_kernel(sigma[2])
    gc = conv(img,g3)
    gc = np.asarray(gc)
    
    g4 = gau_kernel(sigma[3])
    gd = conv(img,g4)
    gd = np.asarray(gd)
    
    g5 = gau_kernel(sigma[4])
    ge = conv(img,g5)
    ge = np.asarray(ge)
    
    dog1 = gb-ga
    
    dog2 = gc-gb
    
    dog3 = gd-gc
    
    dog4 = ge-gd
    
    oct = [dog1,dog2,dog3,dog4]
    return oct

sigma1=[0.707107,1.000000,1.414214,2.000000,2.828427]
sigma2=[1.414214,2.000000,2.828427,4.000000,5.656854]
sigma3=[2.828427,4.000000,5.656854,8.000000,11.313708]
sigma4=[5.656854,8.000000,11.313708,16.000000,22.627417]


pad_img_copy = [pad_img]

resized_pad_img = pad_img[::2,::2]
pad_img_copy.append(resized_pad_img)

resized_pad_img = resized_pad_img[::2,::2]
pad_img_copy.append(resized_pad_img)

resized_pad_img = resized_pad_img[::2,::2]
pad_img_copy.append(resized_pad_img)

#Calculating octaves i.e 1,2,3 and 4; Returned value is a list of difference of gaussian's 1,2,3 and 4
oct1 = calculate_octave(pad_img_copy[0],sigma1)
oct2 = calculate_octave(pad_img_copy[1],sigma2)
oct3 = calculate_octave(pad_img_copy[2],sigma3)
oct4 = calculate_octave(pad_img_copy[3],sigma4)


#Detects Keypoints and returns a list of keypoints for a particular octave
def detect(oct,oct_num,scale_factor):
    #Scale factor is to multiply the co-ordinates by a scaling factor inorder to plot those keypoints got from resized image onto the original image
    points = []
    img = oct[oct_num]
    for i in range(3,len(img) - 8, 1):
        for j in range(3,len(img[0]) - 8, 1):
            mid = img[i+1][j+1]
            check = neighbor(oct,i+1,j+1,oct_num)
            min = True
            max = True
            for n in check:
                if n >= mid:
                    max = False
                if n <= mid:
                    min = False
            if (max or min):
                    points.append([(i+1)*scale_factor,(j+1)*scale_factor])
    return points

#Use to check the current, previous and the next pixel values of respective difference of gaussian's
def neighbor(octave,x,y,oct_num):
    #oct_num is to indicate which should be the current difference of gaussian
    img = octave[oct_num]
    neighbor = [img[x-1,y],
                img[x+1,y],
                img[x,y+1],
                img[x,y-1],
                img[x+1,y+1],
                img[x+1,y-1],
                img[x-1,y+1],
                img[x-1,y-1]]

    prev = octave[oct_num - 1]
    neighbor+=[prev[x,y],
               prev[x+1,y],
               prev[x-1,y],
               prev[x,y+1],
               prev[x,y-1],
               prev[x+1,y+1],
               prev[x+1,y-1],
               prev[x-1,y+1],
               prev[x-1,y-1]]
    
    next = octave[oct_num + 1]
    neighbor+=[next[x,y],
               next[x+1,y],
               next[x-1,y],
               next[x,y+1],
               next[x,y-1],
               next[x+1,y+1],
               next[x+1,y-1],
               next[x-1,y+1],
               next[x-1,y-1]]
    return neighbor

#Calculating respective keypoints for octave 1, octave 2, octave 3 and octave 4
point11 = detect(oct1,1,1) #Parameters are Octave, Current DOG, Scale Factor
point12 = detect(oct1,2,1)
point21 = detect(oct2,1,2)
point22 = detect(oct2,2,2)
point31 = detect(oct3,1,4)
point32 = detect(oct3,2,4)
point41 = detect(oct4,1,8)
point42 = detect(oct4,2,8)

#Storing all keypoints from Octave 1,2,3,4 in array named 'points'
points=[]
points+=point11
points+=point12
points+=point21
points+=point22
points+=point31
points+=point32
points+=point41
points+=point42

#Finding coordinates of the five left-most detected keypoints when the origin is set to be the top-left corner
sorted_points = []
for i in range(len(points)):
    x = points[i][0]
    y = points[i][1]
    euc_dis = (x**2 + y**2)**1/2 #Calculating Euclidean Distance with respect to origin
    sorted_points.append([x,y,euc_dis])
    img[x,y] = 255
 
#Plotting keypoints on the GrayScale image        
cv2.imshow("Keypoints.jpg",img)
cv2.waitKey(0)
        
sorted_points = sorted(sorted_points,key=lambda l:l[2], reverse=False)
print("five left-most detected keypoints: ")
for i in range(5):
    x = sorted_points[i][0]
    y = sorted_points[i][1]
    print(x,y)
