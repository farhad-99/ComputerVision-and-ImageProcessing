import cv2 
import numpy as np
from matplotlib import pyplot as plt
import copy
img = cv2.imread('Greek-ship.jpg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
template = cv2.imread('patch.png',0)
tmp2 = copy.deepcopy(template)
xt, yt  = template.shape
xi , yi  = img2.shape
h, w  = template.shape
#img2 = img2 - np.sum(img2)/(xi * yi)
meantmp = np.sum(tmp2)/(xt * yt)
normtmp = tmp2 - meantmp 
# All the 6 methods for comparison in a list

#MeanImagePatch = cv2.filter2D(img2.astype('float64'),-1,np.ones((xt,yt)))/(xt*yt) 
#normImg = img2 - MeanImagePatch
num = cv2.filter2D(img2.astype('float64'),-1,normtmp.astype('float64')) 
#num2 =cv2.filter2D(normImg.astype('float64'),-1,normtmp.astype('float64'))
#num = num1-num2
#denum1 = np.sum(np.power(normtmp,2)) 
#denum2 = cv2.filter2D(np.power(normImg,2),-1,np.ones((xt,yt)))
#denum = np.power(denum1 * denum2,0.5)
res = num/np.max(num)
cv2.imwrite('ZMCC.jpg',res*255)
#res2 = num2 / denum
sobel = np.array(([1,0,-1],[2,0,-2],[1,0,-1]))
newres = cv2.filter2D(res*255,-1,sobel)
newres = newres/np.max(newres)
threshold =0.3
newres = np.transpose(newres)
loc = np.where( newres >= threshold)
newres = np.transpose(newres)
i=0
point = []
xtmp = 0
ytmp = 0

for pt in zip(*loc[::-1]):
    
    
    if abs(pt[1]-xtmp) >w/3 and abs(pt[0]-ytmp)>h/4:
        cv2.rectangle(img, (pt[1]-int(w/2),pt[0]-int(h/2)), (pt[1]+int(w/2),pt[0]+int(h/2)), (0,0,255), 2)
        xtmp = pt[1]
        ytmp = pt[0]
        i=i+1
    
    
    


cv2.imwrite('founded patches.jpg',newres*255)
cv2.imwrite('res15.jpg',img)


c =0