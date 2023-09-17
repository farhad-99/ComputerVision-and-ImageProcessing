# importing required libraries of opencv
import cv2
import numpy as np
from matplotlib import pyplot as plt

#------- functions ----------
def LogTransformation(I,alpha):
    Enhance_log = np.log10(1 + I * alpha) * 255 / np.log10(1 + 255 * alpha)
    Enhance_log.round()
    Enhance_log = Enhance_log.astype(np.uint8)
    return Enhance_log

def PowerLawTransformation(I,gama):

    Enhance_power = np.power(I/255, gama)*255
    Enhance_power.round()
    Enhance_power = Enhance_power.astype(np.uint8)
    return Enhance_power
def CDF(hist):

    x,y = hist.shape
    cdf = hist*0
    cdf[0] = hist[0]
    for i in range(1,x):
        cdf[i] = cdf[i-1] + hist[i]
    return cdf
def hist_equlize(I):
    HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    S = HSV[:,:,2]
    x,y = S.shape
    S_hist = cv2.calcHist([S], [0], None, [256], [0, 256]) /(x*y)

    union = np.ones((256,1))/256
    S_cdf = CDF(S_hist)
    union_cdf = CDF(union)
    newS = 0 * S
    for i in range(256):
        cdf = S_cdf[i]
        index = np.nonzero(union_cdf > cdf)
        try:
            val = min(index[0])
            index = np.nonzero(S == i)
            tmp = np.ones((x, y))
            newS[index] = val * tmp[index]
        except ValueError:
            s = 5
    HSV[:,:,2]=newS
    newI =cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    return newI

#------main code---------
I = cv2.imread('Enhance1.JPG')


#log enhance
alpha = 0.3;
Enhance_log = LogTransformation(I,alpha)
cv2.imwrite('log_enhance1.jpg', Enhance_log)
###
gama = 0.4;
Enhance_power = PowerLawTransformation(I,gama)
cv2.imwrite('power_enhance1.jpg', Enhance_power)
###
##

res = hist_equlize(I)
HSV1 = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
HSV2 = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
V1 = HSV1[:,:,2]
V2 = HSV2[:,:,2]
'''V2 = cv2.equalizeHist(V1)
HSV2 = 0*HSV1
HSV2[:,:,2] = V2
res = cv2.cvtColor(HSV2, cv2.COLOR_HSV2BGR)'''
V1_hist = cv2.calcHist([V1],[0],None,[256],[0,255])
V2_hist = cv2.calcHist([V2],[0],None,[256],[0,255])
fig, ax = plt.subplots(1, 2)
fig.suptitle('Histogram Compare')
ax[0].plot(V1_hist/max(V1_hist))
ax[0].set_title('Source Image')
ax[1].plot(V2_hist/max(V2_hist))
ax[1].set_title('Enhanced Image')
plt.show()
cv2.imwrite('res01.jpg', res)




