import cv2
import numpy as np
from copy import deepcopy
import skimage.segmentation
from skimage import morphology
import time



start = time.time()
src = cv2.imread('slic.jpg')
lab = cv2.cvtColor(src,cv2.COLOR_BGR2Lab)
L,a,b = cv2.split(lab)
img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
dx = np.array(([1,0,-1],[1,0,-1],[1,0,-1]))
dy = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
img_x = cv2.filter2D(img,cv2.CV_8U,dx)
img_y = cv2.filter2D(img,cv2.CV_8U,dy)
img_grad = img_x**2 + img_y**2

H_,W_ = img.shape[0:2]
ratio = H_/W_

K_=[(12,64),(10,256),(6,1024),(4,2048)]
numres = 6
for iterate,k in K_:
    
    alpha = 0.02
    nw = round((k/(0.75))**0.5 )
    nh = round(0.75*nw)
    x = nw*nh

    center_h = np.linspace(0,H_,nh+1,dtype='int')[1:nh+1]
    center_w = np.linspace(0,W_,nw+1,dtype='int')[1:nw+1]
    steph = center_h[1]-center_h[0]
    stepw = center_w[1]-center_w[0]
    step = max(steph,stepw)
    i=0
    j=0
    centers_ = np.zeros((nh,nw,5),dtype=int)
    #centerh_ = centerw_*0
    for j in range(nh):
        for i in range(nw):
            cntrw = center_w[i]
            cntrh = center_h[j]
            window = img_grad[cntrh-2:cntrh+2,cntrw-2:cntrw+2]
            result = np.where(window == np.amin(window))
            res = zip(*result[::-1])
            for loc in res:
                w = cntrw+loc[0] -2
                h = cntrh+loc[1] -2
                
                centers_[j,i,0] = w
                centers_[j,i,1] = h
                centers_[j,i,2] = L[h,w]
                centers_[j,i,3] = a[h,w]
                centers_[j,i,4] = b[h,w]


    Label = a*0
    number =4

    i=4000
    j=3000

    tmp = np.zeros((400,400,3),dtype=np.uint8)
    tmp =lab [0:400,0:400,:]
    d = np.zeros((H_,W_))


    position = np.indices((H_,W_)).reshape(2,H_,W_)
    for iterate_ in range(iterate):
        label = np.zeros((H_,W_))
        mindist = 999999*np.ones((H_,W_))

        for i in range(0,nh):
            for j in range(0,nw):
                w1 = max(centers_[i,j,0]-step,0)
                w2 = min(centers_[i,j,0]+step,W_)
                h1 = max(centers_[i,j,1]-step,0)
                h2 = min(centers_[i,j,1]+step,H_)
                w_ = centers_[i,j,0]
                h_ = centers_[i,j,1]
                l_ = centers_[i,j,2]
                a_ = centers_[i,j,3]
                b_ = centers_[i,j,4]
                dlab = np.zeros((h2-h1,w2-w1))
                dxy = dlab *0
                dcalc = dlab*0
                dcurrent = deepcopy(mindist[h1:h2,w1:w2])
                labelcurrent = deepcopy(label[h1:h2,w1:w2])
                dlab = (lab[h1:h2,w1:w2,0]-l_)**2 + (lab[h1:h2,w1:w2,1]-a_)**2 + (lab[h1:h2,w1:w2,2]-b_)**2
                dxy = (position[0,h1:h2,w1:w2]-h_)**2 + (position[1,h1:h2,w1:w2]-w_)**2
                newlabel = (i*nw+j)*np.ones((h2-h1,w2-w1))
                dcalc = dlab + alpha*dxy
                mask_findmin = dcalc < dcurrent
                
                indx = np.where(mask_findmin == True)
                dcurrent[indx] = dcalc[indx]
                labelcurrent[indx] = newlabel[indx]
                
                #cw,ch = find_best(i,j,centers_)
                #Label[j:min(j+10,H_-1),i:min(i+10,W_-1)] = ch*nw+cw
                mindist[h1:h2,w1:w2] = dcurrent
                label[h1:h2,w1:w2] = labelcurrent

              
            
        hmat = position[0,:,:]
        wmat = position[1,:,:]
        lmat,amat,bmat = cv2.split(lab)
        for i in range(nh):
            for j in range(nw):
                
                indx = np.where(label == (i*nw + j))
                centers_[i,j,0]  = np.average(wmat[indx])
                centers_[i,j,1]  = np.average(hmat[indx])
                centers_[i,j,2]  = np.average(lmat[indx])
                centers_[i,j,3]  = np.average(amat[indx])
                centers_[i,j,4]  = np.average(bmat[indx])
    
    label=morphology.closing(label,np.ones((20,20)))
    boundaries = skimage.segmentation.find_boundaries(label)
    indx = np.where(boundaries == True)
    zero = np.zeros((H_,W_))
    B,G,R = cv2.split(src)
    B[indx] = zero[indx]
    G[indx] = zero[indx]
    R[indx] = zero[indx]
    res = cv2.merge((B,G,R))
    cv2.imwrite('res0'+str(numres)+'.jpg',res)
    numres +=1
    print('process time for k= '+str(k)+':',time.time()-start)


x=5