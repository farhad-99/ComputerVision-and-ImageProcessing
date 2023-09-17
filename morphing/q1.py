import cv2
import numpy as np
from scipy.spatial import Delaunay
import skimage.segmentation
from skimage import morphology
import os
'''gw1 = np.zeros((30,1))
gh1 = np.zeros((30,1))
gw2 = np.zeros((30,1))
gh2 = np.zeros((30,1))
i=0
j=0
def find_pos1(event,x,y,flags,param):
    global gw1,gh1,i
    if event == cv2.EVENT_LBUTTONDOWN:
        
        gw1[i]= x
        gh1[i] =y
        print (gw1[i],gh1[i])
        i=i+1
def find_pos2(event,x,y,flags,param):
    global gw2,gh2,j
    if event == cv2.EVENT_LBUTTONDOWN:
        
        gw2[j]= x
        gh2[j] =y
        print (x,gh2[j])
        j=j+1
        
im1 = cv2.imread('res01.jpg')
cv2.namedWindow('im1',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('im1',find_pos1)
cv2.imshow('im1',im1)
cv2.waitKey()
cv2.destroyAllWindows() 
im2 = cv2.imread('res02.jpg')
cv2.namedWindow('im2',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('im2',find_pos2)
cv2.imshow('im2',im2)
cv2.waitKey()
cv2.destroyAllWindows() 
txt1= open("im1.txt","a")
for i in range(len(gw1)):
    txt1.write(str(int(gw1[i])) + " " + str(int(gh1[i])) + "\n")
txt1.close()
txt2= open("im2.txt","a")
for i in range(len(gw2)):
    txt2.write(str(int(gw2[i])) + " " + str(int(gh2[i])) + "\n")
txt2.close()'''
im1 = cv2.imread('res01.jpg')
im2 = cv2.imread('res02.jpg')
H_,W_ = im1.shape[0:2]
frames = 45
with open('im1.txt') as f:
   data = []
   rows=30
   cols = 2
   for i in range(0, rows):
      data.append(list(map(float, f.readline().split()[:cols])))

points1 = np.array(data)
w1 = points1[:,0]
h1 = points1[:,1]

with open('im2.txt') as f:
   data = []
   rows=30
   cols = 2
   for i in range(0, rows):
      data.append(list(map(float, f.readline().split()[:cols])))

points2 = np.array(data)
w2 = points2[:,0]
h2 = points2[:,1]
tri_ = Delaunay(points1)
for i in range(frames):

    w_current = np.int32(w1 + (w2-w1)*i/frames)
    h_current = np.int32(h1 + (h2-h1)*i/frames)
    #mask_src = im1*0
    #mask_dst = im1*0
    res = im1*0
    for tri in tri_.simplices:
        mask = im1*0
        p1 = tri[0]
        p2 = tri[1]
        p3 = tri[2]
        src = np.array(([w1[p1],h1[p1]] , [w1[p2],h1[p2]] ,[w1[p3],h1[p3]] )).astype(np.float32) 
        dst = np.array(([w2[p1],h2[p1]] , [w2[p2],h2[p2]] ,[w2[p3],h2[p3]] )).astype(np.float32)
        current = np.array(([w_current[p1],h_current[p1]] , [w_current[p2],h_current[p2]] ,[w_current[p3],h_current[p3]] )).astype(np.float32)
        warp_src = cv2.getAffineTransform(src,current)
        warp_dst = cv2.getAffineTransform(dst,current)
        mask = cv2.fillConvexPoly(mask, current.astype(np.int32) , (1,1,1))
        warped_src = cv2.warpAffine(im1,warp_src,(W_,H_))
        warped_dst = cv2.warpAffine(im2,warp_dst,(W_,H_))
        nonzero = res ==0
        '''label = mask[:,:,0]
        boundaries = skimage.segmentation.find_boundaries(label)
        indx = np.where(boundaries == True)
        boundry_val = ((1-i/frames) * warped_src + (i/frames) * warped_dst)'''
        res = res + ((1-i/frames) * warped_src + (i/frames) * warped_dst) * mask * nonzero
        if i==14 or i==29:
           cv2.imwrite('res%02d.jpg'%((i+1)/15+2),res) 
    cv2.imwrite('out%02d.jpg'%(i+1),res)
    cv2.imwrite('out%02d.jpg'%(90-i),res) 
    
    
        #cv2.fillConvexPoly(mask_src, np.int32(src), (2*i, 2*i, 2*i))
        #cv2.fillConvexPoly(mask_dst, np.int32(src), (2*i, 2*i, 2*i))
        

'''label = mask_src[:,:,0]
label = mask_dst[:,:,0]
boundaries = skimage.segmentation.find_boundaries(label)
indx = np.where(boundaries == True)
zero = np.zeros((H_,W_))
B,G,R = cv2.split(im1)
B,G,R = cv2.split(im2)
B[indx] = zero[indx]
G[indx] = zero[indx]
R[indx] = zero[indx]
res = cv2.merge((B,G,R))
cv2.imwrite('tri.jpg',res)'''
os.system('ffmpeg -framerate 15/1 -i out%02d.jpg -r 15 -start_number 1 -vframes 90 -vcodec mpeg4 morph.mp4')
z=5