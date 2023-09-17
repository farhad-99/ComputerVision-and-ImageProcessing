import numpy as np
import cv2
import math
import cmath
from matplotlib import pyplot as plt
#--------- functions
gwnear = np.zeros((10,1))
ghnear = np.zeros((10,1))
inear=0
gwfar = np.zeros((10,1))
ghfar = np.zeros((10,1))
ifar=0
def gaussian_filt(sigma,x,y):
    
    xmid = int(x/2)
    ymid = int(y/2)
    g = np.zeros((2*xmid+1,2*ymid+1))
    for i in range(-xmid,xmid+1):
        for j in range(-ymid,ymid+1):
            
             g[i+xmid, j+ymid] = math.exp(-(i**2+j**2)/(2*sigma**2) )/(2*math.pi*sigma**2)
    if(x%2 == 0):
        g = np.concatenate((g[:xmid,:],g[xmid+1:,:]),0)
    if(y%2 == 0):
        g = np.concatenate((g[:,:ymid],g[:,ymid+1:]),1)
    tmp = np.sum(g)
    g = g / tmp
    return g

def fft2(V):
    Vfourier1 = np.fft.fft2(V)
    Vfourier1 = np.fft.fftshift(Vfourier1)
    return Vfourier1
def ifft2(Vfourier):
    newVfourier = np.fft.ifftshift(Vfourier)
    newV = np.fft.ifft2(newVfourier)
    newV = np.real(newV)   
    return newV 

# mouse callback function

def find_posnear(event,x,y,flags,param):
    global gwnear,ghnear,inear
    if event == cv2.EVENT_LBUTTONDOWN:
        
        gwnear[inear]= x
        ghnear[inear] =y
        inear=inear+1
def find_posfar(event,x,y,flags,param):
    global gwfar,ghfar,ifar
    if event == cv2.EVENT_LBUTTONDOWN:
        
        gwfar[ifar]= x
        ghfar[ifar] =y
        ifar=ifar+1        
#----------main--------
img1 = cv2.imread('res19-near.jpg')
cv2.namedWindow('near',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('near',find_posnear)
cv2.imshow('near',img1)
cv2.waitKey()
img2 = cv2.imread('res20-far.jpg')
cv2.namedWindow('far',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('far',find_posfar)
cv2.imshow('far',img2)
cv2.waitKey()
src = np.array(([gwnear[0],ghnear[0]] , [gwnear[1] , ghnear[1]] ,[gwnear[2],ghnear[2]], [gwnear[3],ghnear[3]] , [gwnear[4],ghnear[4]] , [gwnear[5],ghnear[5]], [gwnear[6],ghnear[6]], [gwnear[7],ghnear[7]] )).astype(np.float32)
dst = np.array(([gwfar[0],ghfar[0]] , [gwfar[1] , ghfar[1]] ,[gwfar[2],ghfar[2]], [gwfar[3],ghfar[3]], [gwfar[4],ghfar[4]], [gwfar[5],ghfar[5]], [gwfar[6],ghfar[6]], [gwfar[7],ghfar[7]] )).astype(np.float32)
warp_mat , stat = cv2.findHomography(src,dst)
warpednear = cv2.warpPerspective(img1, warp_mat, (img2.shape[1],img2.shape[0]))
cv2.imwrite('res21-near.jpg',warpednear)
cv2.imwrite('res22-near.jpg',img2)

img1 = cv2.imread('res21-near.jpg') #near
img2 = cv2.imread('res22-near.jpg')    #far
HSV1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
V1 = HSV1[:,:,2]
HSV2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
V2 = HSV2[:,:,2]
#images fourier

Vfourier1 = fft2(V1)
plt.imshow(np.log(np.abs(Vfourier1)),cmap = 'gray')
plt.title('dft-near'), plt.xticks([]), plt.yticks([])
plt.savefig("res23-dft-near.jpg")
#plt.show() 

Vfourier2 =fft2(V2)

plt.imshow(np.log(np.abs(Vfourier2)),cmap = 'gray')
plt.title('dft-far'), plt.xticks([]), plt.yticks([])
plt.savefig("res24-dft-far.jpg")
#plt.show() 


#---filters

x,y = img1[:,:,1].shape
sigmar =1 #highpass filter
sigmas = 3 #lowpass filter
#high pass filter
lowpass1 = gaussian_filt(sigmar,x,y)
Hlp1 = fft2(lowpass1)
Hlp1 = abs(Hlp1)
Hhp = 1 - Hlp1
plt.imshow(Hhp,cmap = 'gray')
plt.title('highpass-1'), plt.xticks([]), plt.yticks([])
plt.savefig("res25-highpass-1.jpg")
#plt.show() 
#low pass filter
lowpass2 = gaussian_filt(sigmas,x,y)
Hlp = fft2(lowpass2)
Hlp = np.abs(Hlp)
plt.imshow(Hlp,cmap = 'gray')
plt.title('lowpass-3'), plt.xticks([]), plt.yticks([])
plt.savefig("res26-lowpass-3.jpg")
#plt.show() 

#----filtering

HighpassimgV = Vfourier1 * Hhp
plt.imshow(20*np.log(np.abs(HighpassimgV)),cmap = 'gray')
plt.title('highpassed'), plt.xticks([]), plt.yticks([])
plt.savefig("res27-highpassed.jpg")
#plt.show() 
LowpassimgV = Vfourier2 * Hlp
plt.imshow(20*np.log(np.abs(LowpassimgV)),cmap = 'gray')
plt.title('lowpassed'), plt.xticks([]), plt.yticks([])
plt.savefig("res28-lowpassed.jpg")
#plt.show() 

B1Fourier = fft2(img1[:,:,0]) * Hhp
G1Fourier = fft2(img1[:,:,1]) * Hhp
R1Fourier = fft2(img1[:,:,2]) * Hhp

B2Fourier = fft2(img2[:,:,0]) * Hlp
G2Fourier = fft2(img2[:,:,1]) * Hlp
R2Fourier = fft2(img2[:,:,2]) * Hlp


HybridB = B1Fourier + B2Fourier
HybridG = G1Fourier + G2Fourier
HybridR = R1Fourier + R2Fourier

Hybrid = LowpassimgV + HighpassimgV

plt.imshow(20*np.log(np.abs(Hybrid)),cmap = 'gray')
plt.title('Hybrid'), plt.xticks([]), plt.yticks([])
plt.savefig("res29-Hybrid.jpg")
#plt.show() 


hybridb = ifft2(HybridB)
hybridg = ifft2(HybridG)
hybridr = ifft2(HybridR)
newbgr = np.zeros((x,y,3))
newbgr[:,:,0] = hybridb
newbgr[:,:,1] = hybridg
newbgr[:,:,2] = hybridb

cv2.imwrite('res30-hybrid-near.jpg' , newbgr)
new2 = cv2.resize(newbgr, (int(0.3*y), int(0.3*x)), interpolation = cv2.INTER_AREA)
cv2.imwrite('res31-hybrid-far.jpg' , new2)
test =4