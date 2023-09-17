import cv2
import numpy as np

gw0 = np.zeros((12,1))
gh0 = np.zeros((12,1))
i=0
# mouse callback function
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
def find_pos(event,x,y,flags,param):
    global gw0,gh0,i
    if event == cv2.EVENT_LBUTTONDOWN:
        
        gw0[i]= x
        gh0[i] =y
        print (gw0[i],gh0[i])
        i=i+1
        
def FindHomography(gw,gh):
    w1b1 = int (((gw[0] - gw[1]) ** 2 + (gh[0] - gh[1]) ** 2 ) **0.5) +1
    w2b1 = int (((gw[3] - gw[2]) ** 2 + (gh[3] - gh[2]) ** 2 ) **0.5) +1
    h1b1 = int (((gw[2] - gw[1]) ** 2 + (gh[2] - gh[1]) ** 2 ) **0.5) +1
    h2b1 = int (((gw[3] - gw[0]) ** 2 + (gh[3] - gh[0]) ** 2 ) **0.5) +1

    wb1 = min(w1b1 , w2b1)
    hb1 = min(h1b1 , h2b1)

    src = np.array(([gw[0],gh[0]] , [gw[1] , gh[1]] ,[gw[2],gh[2]], [gw[3],gh[3]] )).astype(np.float32)
    dst = np.array (([wb1-1,hb1-1] ,[0,hb1-1] , [0,0],[wb1-1,0] )).astype(np.float32)
    warp_mat , stat = cv2.findHomography(src,dst)
    return wb1,hb1,warp_mat

def Warping(img,warp_mat,wb1,hb1):
    mat_inverse = np.linalg.inv(warp_mat)
    
    warped = np.zeros((hb1,wb1,3))
    for i in range (wb1):
        for j in range(hb1):
            vec1 = np.array((i,j,1))
            vec2 = np.matmul(mat_inverse , vec1)
            w = (vec2[0]/vec2[2])
            h = (vec2[1]/vec2[2])
            a = h-int(h)
            b = w - int(w)
            for k in range(3):
                f00 = img[int(h) , int(w) , k]
                f01 = img[int(h) , int(w)+1 , k]
                f10 = img[int(h)+1 , int(w) , k]
                f11 = img[int(h)+1 , int(w) +1, k]
                m1 = np.array(([1-a ,a]))
                m2 = np.array(([f00 , f01],[f10 , f11]))
                m3 = np.array((1-b , b))
                m12 = np.matmul(m1,m2)
                warped[j,i,k] = np.matmul(m12,m3)
    #warped = cv2.warpPerspective(img, warp_mat, (wb1,hb1))
    return warped

# Create a black image, a window and bind the function to window

img = cv2.imread('books.jpg')
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',find_pos)


		

cv2.imshow('image',img)
cv2.waitKey()



gw1 = gw0[0:4]
gh1 = gh0[0:4]

wb1,hb1,warp_mat1 = FindHomography(gw1,gh1)
warped1 = Warping(img,warp_mat1,wb1,hb1)
cv2.imwrite('res16.jpg',warped1)

gw2 = gw0[4:8]
gh2 = gh0[4:8]

wb2,hb2,warp_mat2 = FindHomography(gw2,gh2)
warped2 = Warping(img,warp_mat2,wb2,hb2)
cv2.imwrite('res17.jpg',warped2)

gw3 = gw0[8::]
gh3 = gh0[8::]

wb3,hb3,warp_mat3 = FindHomography(gw3,gh3)
warped3 = Warping(img,warp_mat3,wb3,hb3)
cv2.imwrite('res18.jpg',warped3)


cv2.destroyAllWindows()



