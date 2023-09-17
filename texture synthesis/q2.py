import cv2
import numpy as np
from random import random , randint
import copy
def rand_patch(txt,l):
    x,y = txt.shape[0:2]
    i = randint(0,x-l)
    j = randint(0,y-l)
    return txt[i:i+l,j:j+l,:]

def random_point(res,threshold):
    loc = np.where( res <= threshold)
    point =[]
    for pt in zip(*loc[::-1]):        
        point.append(pt)
    
    point = np.array(point)
    l = point.shape[0]
    rndnext = randint(0,l-1)
    y = point[rndnext,0]
    x = point[rndnext,1]
    return x,y

def find_patch(tmp,patch,mask,l,b):
    patch1gray = cv2.cvtColor(patch.astype('uint8'),cv2.COLOR_BGR2GRAY)
    xt,yt = tmp.shape[0:2]
    
    txtgray = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(txtgray[0:xt-l-b,0:yt-l-b],patch1gray,cv2.TM_SQDIFF,None,mask)
    res = res/np.max(res)
    min = np.min(res)
    x,y = random_point(res,min+0.1)
    return x,y


def left_right_border(patch1gray,patch2gray):
    border = (patch2gray - patch1gray)**2 
    x,b = border.shape
    cost = np.zeros((x,b))
    cost[x-1,:] = border[x-1,:]

    for k in range(x-2,-1,-1):
        for j in range(b):
            cost[k,j] = border[k,j] + min(cost[k+1,max(j-1,0)],cost[k+1,j],cost[k+1,min(j+1,b-1)])

    row1 = cost[0,:]
    c1 = []
    ctmp = np.argwhere(cost[0,:]== min(row1))[0]
    ctmp = ctmp[0]
    c1.append(ctmp)
    for k in range(1,x):
        row=cost[k,max(ctmp-1,0):min(ctmp+1,b-1)+1]
        lenx = row.shape[0]
        #print(lenx)
        minrow = np.min(row)
        ctmp0 = np.argwhere(row== (minrow))[0]
        if(lenx == 3):
            ctmp = ctmp0[0] + (ctmp-1)
        elif(lenx == 2):
            if(ctmp==b-1):
               ctmp = ctmp0[0] + ctmp-1
            else:
                ctmp = ctmp0[0] + ctmp
        c1.append(ctmp)
    return c1
    


def up_down_border(patch1gray,patch2gray):
    border = (patch2gray - patch1gray)**2 
    b,y = border.shape
    cost = np.zeros((b,y))
    cost[:,y-1] = border[:,y-1]

    for k in range(y-2,-1,-1):
        for j in range(b):
            cost[j,k] = border[j,k] + min(cost[max(j-1,0),k+1],cost[j,k+1],cost[min(j+1,b-1),k+1])
    column1 = cost[:,0]
    r1=[]
    rtmp = np.argwhere(cost[:,0]== min(column1))[0]
    rtmp = rtmp[0]
    r1.append(rtmp)
    
    for k in range(1,y):
        col=cost[max(rtmp-1,0):min(rtmp+1,b-1)+1,k]
        leny = col.shape[0]
        
        mincol = np.min(col)
        rtmp0 = np.argwhere(col== (mincol))[0]
        if(leny == 3):
            rtmp = rtmp0[0] + (rtmp-1)
        elif(leny == 2):
            if(rtmp==b-1):
               rtmp = rtmp0[0] + rtmp-1
            else:
                rtmp = rtmp0[0] + rtmp
        r1.append(rtmp)
    return r1


l=100
b=30
size = 2500
texture1 = cv2.imread('texture01.jpg')
#texture1 = cv2.imread('texture11.jpeg')
#texture1 = cv2.imread('txt02.jpg')
#texture1 = cv2.imread('txt01.jpg')
src = copy.deepcopy(texture1)
txtgray = cv2.cvtColor(texture1,cv2.COLOR_BGR2GRAY)
xt,yt = texture1.shape[0:2]

final = np.zeros((size,size,3))
final[0:l:,0:l,:] = rand_patch(texture1,l)
mask = np.ones((l,b),dtype = np.uint8)


for i in range(l,size,l):
    patch = final[0:l:,i-b:i,:]
    patch1gray = cv2.cvtColor(patch.astype('uint8'),cv2.COLOR_BGR2GRAY)
    
    x,y = find_patch(texture1,patch,mask,l,b)
    cv2.rectangle(src, (y,x), (y + l, x + l), (0,0,255), 2)
    final[0:l,i:i+l,:] = texture1[x:x+l,y+b:y+l+b,:]
    
    patch2 = texture1[x:x+l,y:y+b,:] 
    patch2gray = cv2.cvtColor(patch2,cv2.COLOR_BGR2GRAY)

    c1 = left_right_border(patch1gray,patch2gray)

    for k in range(l):
        final[k,i-b+c1[k]:i,:] = patch2[k,c1[k]:b,:]


mask = np.ones((b,l),dtype = np.uint8)



for i in range(l,size,l):
    patch = final[i-b:i,0:l,:]
    patch1gray = cv2.cvtColor(patch.astype('uint8'),cv2.COLOR_BGR2GRAY)
    
    x,y = find_patch(texture1,patch,mask,l,b)
    final[i:i+l,0:l,:] = texture1[x+b:x+l+b,y:y+l,:]
    cv2.rectangle(src, (y,x), (y + l, x + l), (0,255,0), 2)
    patch2 = texture1[x:x+b,y:y+l,:] 
    patch2gray = cv2.cvtColor(patch2,cv2.COLOR_BGR2GRAY)

    r1 = up_down_border(patch1gray,patch2gray)
   
    for k in range(l):
        final[i-b+r1[k]:i,k,:] = patch2[r1[k]:b,k,:]


mask = np.ones((l+b,l+b),dtype = np.uint8)
mask[b:l+b,b:l+b] = np.zeros((l,l))



for i in range(l,size,l):
    for j in range(l,size,l):
        
        patch = copy.deepcopy(final[i-b:i+l,j-b:j+l,:])
        patch1gray = cv2.cvtColor(patch.astype('uint8'),cv2.COLOR_BGR2GRAY,0)
    
        x,y = find_patch(texture1,patch,mask,l,b)
        
        cv2.rectangle(src, (y,x), (y + l, x + l), (255,0,255), 2)

        final[i:i+l,j:j+l,:] = texture1[x+b:x+l+b,y+b:y+l+b,:]

        patch_up = final[i-b:i,j-b:j+l,:]
        patch_down = texture1[x:x+b,y:y+b+l,:]
        patch_upgray = cv2.cvtColor(patch_up.astype('uint8'),cv2.COLOR_BGR2GRAY)
        patch_downgray = cv2.cvtColor(patch_down,cv2.COLOR_BGR2GRAY)
        r1 = up_down_border(patch_upgray,patch_downgray)
        dj=b-r1[0]

        gg = copy.deepcopy(final[i-b:i,j-b:j,:])
        
        
        patch_left = final[i-b:i+l,j-b:j,:]
        patch_right = texture1[x:x+l+b,y:y+b,:]
        patch_leftgray = cv2.cvtColor(patch_left.astype('uint8'),cv2.COLOR_BGR2GRAY)
        patch_rightgray = cv2.cvtColor(patch_right,cv2.COLOR_BGR2GRAY)
        c1 = left_right_border(patch_leftgray,patch_rightgray)
        
        for k in range(c1[0],b):
            gg[r1[k]:b,k,:] = patch_down[r1[k]:b,k,:]
            
        for k in range (r1[0],b):
            gg[k,0:c1[k],:] = copy.deepcopy(final[i-b+k,j-b:j-b+c1[k],:])
            gg[k,c1[k]:b,:] = patch_right[k,c1[k]:b,:]

        for k in range(b,l+b,1):
                final[i-b+r1[k]:i,j-b+k,:] = patch_down[r1[k]:b,k,:] 

        
        
        

        for k in range(b,l+b,1):
                final[i-b+k,j-b+c1[k]:j,:] = patch_right[k,c1[k]:b,:] 

        final[i-b:i,j-b:j,:] = gg


cv2.imwrite('patch_uesed11.jpg',src)
cv2.imwrite('res11.jpg',final)

#cv2.imwrite('patch_uesed12.jpg',src)
#cv2.imwrite('res12.jpg',final)


#cv2.imwrite('patch_uesed13.jpg',src)
#cv2.imwrite('res13.jpg',final)

#cv2.imwrite('patch_uesed14.jpg',src)
#cv2.imwrite('res14.jpg',final)

test = 2