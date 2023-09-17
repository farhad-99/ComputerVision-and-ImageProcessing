
import cv2
import numpy as np
from math import pi,ceil


def myEdgeFilter(img0,sigma):
    h_size = 2*ceil(3*sigma)+1
    img_gray = cv2.GaussianBlur(img0,(h_size,h_size),sigma)

    H_,W_ = img0.shape[0:2]
    
    imgx = cv2.Sobel(img_gray,cv2.CV_32F,1,0,ksize=5)
    imgy = cv2.Sobel(img_gray,cv2.CV_32F,0,1,ksize=5)


    img_grad = np.hypot(imgx, imgy)
    img_th = np.arctan2(imgy, imgx)*180/pi
    res = img_grad*0
    for i in range(H_-1):
        for j in range(W_-1):
            th = img_th[i,j]
            n1 = 255
            n2 = 255
            step = 22.5
            if th < 0:
                th += 180

            if (0 <= th < 22.5) or (180 -step<= th <= 180 ):#map to 0
                n1 = img_grad[i, j+1]
                n2 = img_grad[i, j-1]
                if (img_grad[i,j] >= n1) and (img_grad[i,j] >= n2):
                    res[i,j] = img_grad[i,j]
                else :
                    res[i,j] =0
            elif (22.5 <= th < 67.5):   #map to 45
                n1 = img_grad[i+1, j-1]
                n2 = img_grad[i-1, j+1]
                if (img_grad[i,j] >= n1) and (img_grad[i,j] >= n2):
                    res[i,j] = img_grad[i,j]
                else :
                    res[i,j] =0
            elif (67.5 <= th < 112.5):  #map to 90
                n1 = img_grad[i+1, j]
                n2 = img_grad[i-1, j]
                if (img_grad[i,j] >= n1) and (img_grad[i,j] >= n2):
                    res[i,j] = img_grad[i,j]
                else :
                    res[i,j] =0
            elif (112.5 <= th < 157.5): #map to 135
                n1 = img_grad[i-1, j-1]
                n2 = img_grad[i+1, j+1]
                if (img_grad[i,j] >= n1) and (img_grad[i,j] >= n2):
                    res[i,j] = img_grad[i,j]
                else :
                    res[i,j] =0
    #res[res>=125] = 200
    #res[res<125] = 0
    return res

img0 = cv2.imread('edge_q.jpg')
img_gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
sigma = 6

res = myEdgeFilter(img_gray,sigma)

res_canny = cv2.Canny(img0,25,100)  
cv2.imwrite('res_myEdgeFilter.jpg',res) 
res[res>=125] = 200
res[res<125] = 0     
cv2.imwrite('res_myEdgeFilterThreshold.jpg',res)
cv2.imwrite('res_Canny.jpg',res_canny)
test = 5