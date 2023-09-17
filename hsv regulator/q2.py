import cv2
import numpy as np

   
def nothing(x):
  pass

cv2.namedWindow('Image')
src = cv2.imread("flowers.png")



cv2.createTrackbar('Hue', 'Image',179,3*179,nothing)
cv2.createTrackbar('Sat', 'Image',255,3*255,nothing)
cv2.createTrackbar('Var', 'Image',255,3*255,nothing)
while (True):
    Hue = cv2.getTrackbarPos('Hue', 'Image')/179
    Sat = cv2.getTrackbarPos('Sat', 'Image')/255
    Var = cv2.getTrackbarPos('Var', 'Image')/255
    hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    h = Hue * h
    s = Sat * s
    v = Var * v
    h = np.clip(h,0,179)
    v = np.clip(v,0,255)
    s = np.clip(s,0,255)
    hsv = cv2.merge((h,s,v)).astype('uint8')
    image = cv2.cvtColor(hsv , cv2.COLOR_HSV2BGR)
   
    
    cv2.imshow('Image', image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()