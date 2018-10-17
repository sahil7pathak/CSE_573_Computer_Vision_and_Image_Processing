'''
Title          :- Project 1, Task 1 - Edge Detection
Author         :- Sahil Suhas Pathak
UBITName       :- sahilsuh
Person Number  :- 50289739

'''

#Required imports
import cv2
import numpy as np

#Reading the Image
sample = cv2.imread('task1.png',0)
cv2.imshow('image',sample)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Zero Padding
h, w = sample.shape[:2]
h = h + 2
w = w + 2
pad_img = [[0 for x in range(w)] for y in range(h)]

print(len(sample), len(sample[0]))
print(len(pad_img), len(pad_img[0]))

for i in range(len(sample)):
    for j in range(len(sample[0])):
        pad_img[i+1][j+1] = sample[i][j]
        
pad_img = np.asarray(pad_img)

#Methods
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
def convolution(image, kernel):
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

#Defines the output image, combination of gradient_x and gradient_y
def output(img1, img2):
    h, w = img1.shape
    result = [[0 for x in range(w)] for y in range(h)] 
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            result[i][j] = (img1[i][j]**2 + img2[i][j]**2)**(1/2)
            if(result[i][j] > 255):
                result[i][j] = 255
            elif(result[i][j] < 0):
                result[i][j] =  0
    result = np.asarray(result)            
    return result

#Returns the maximum value from gradient_y/gradient_x
def maximum(gradient):
   max = gradient[0][0]
   for i in range(len(gradient)):
       for j in range(len(gradient[0])):
           if (max < gradient[i][j]):
               max = gradient[i][j]
   return max

#Returns the gradient_y/gradient_x with absolute values
def absolute_value(gradient):
    for i in range(len(gradient)):
        for j in range(len(gradient[0])):
            if(gradient[i][j] < 0):
                gradient[i][j] *= -1
            else:
                continue
    return gradient

#Plotting gradient_y
w, h = 3, 3
kernel_y = [[0 for x in range(w)] for y in range(h)] 
kernel_y = np.asarray(kernel_y)
kernel_y[0,0] = 1
kernel_y[0,1] = 2
kernel_y[0,2] = 1
kernel_y[1,0] = 0
kernel_y[1,1] = 0
kernel_y[1,2] = 0
kernel_y[2,0] = -1
kernel_y[2,1] = -2
kernel_y[2,2] = -1
gradient_y = convolution(sample, kernel_y)
#print(gradient_y)
gradient_y = absolute_value(gradient_y) / maximum(absolute_value(gradient_y))
cv2.imshow("gradient_y",gradient_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Plotting gradient_x
w, h = 3, 3
kernel_x = [[0 for x in range(w)] for y in range(h)] 
kernel_x = np.asarray(kernel_x)
kernel_x[0,0] = 1
kernel_x[0,1] = 0
kernel_x[0,2] = -1
kernel_x[1,0] = 2
kernel_x[1,1] = 0
kernel_x[1,2] = -2
kernel_x[2,0] = 1
kernel_x[2,1] = 0
kernel_x[2,2] = -1
gradient_x = convolution(sample, kernel_x)
#print(gradient_x)
gradient_x = absolute_value(gradient_x) / maximum(absolute_value(gradient_x))
cv2.imshow("gradient_x",gradient_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Plotting final output image
sobel = output(gradient_x, gradient_y)
cv2.imshow("Output_Image", sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()


