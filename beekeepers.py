import cv2
import numpy as np
import argparse

size = (17,17) # размер лабиринта


def cornering(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bi = cv2.bilateralFilter(gray, 5, 75, 75)
    cv2.imshow('bi',bi)
    dst = cv2.cornerHarris(bi, 2, 3, 0.04)#ищем углы по Харрису
    dst = cv2.dilate(dst,None)
    # ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    mask = np.zeros_like(gray)
#--- applying a threshold and turning those pixels above the threshold to white ---
    mask[dst>0.01*dst.max()] = 255

    coor = np.argwhere(mask) #

    coor = np.flip(coor)
    corners = np.zeros((4, 2), dtype="float32")
    s = np.float32(coor).sum(axis=1)
    corners[0]=coor[np.argmin(s)]
    corners[3]=coor[np.argmax(s)]           # ищем углы лабиринта
    d = np.diff(np.float32(coor),axis=1)
    corners[2]=coor[np.argmin(d)]
    corners[1]=coor[np.argmax(d)]
    return corners
# img2 = img.copy()
# for pt in corners:
#     cv2.circle(img2, tuple((pt)), 3, (0, 0, 255), -1)  # посмотреть, что углы верно определились
# cv2.imshow('io',img2)
def persp_transf(corners):
    pts1 = corners
    pts2 = np.float32([[0,0],[20*size[0],0],[0,20*size[1]],[20*size[0],20*size[1]]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    result = cv2.warpPerspective(img,matrix,(20*size[0],20*size[1]))
    # приведение перспективы
    return result

def getting_pretty(res):
    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.imshow('vs',gray)
    im_bw = np.flip(gray,1)

    im_bw = np.rot90(im_bw)
    return im_bw

#cv2.imshow('v',im_bw)

def thresh(im_bw):
    thresh = 160
    im_bw = cv2.threshold(im_bw, thresh, 255, cv2.THRESH_BINARY)[1]
    # blur = cv2.GaussianBlur(im_bw,(5,5),0)
    # im_bw = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return im_bw


#cv2.imshow('s',im_bw)  #ЧБ большая



def reSize(im_bw):
    im_bw = cv2.resize(im_bw, size, interpolation=cv2.INTER_BITS)
    return im_bw# сводим размер к желаемому
#cv2.imshow('p',im_bw)
def bitify(im_bw):
    im_bw = np.where(im_bw > 150, 1, 0)
    nim = np.array(im_bw)
    #print(nim.size)
    h,w=im_bw.shape
# Print that puppy out
    for r in range(h):
        for c in range(w):
            print(nim[r,c],end='')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('img', type=str, help="Image to matrify")
    args = parser.parse_args()
    img = cv2.imread(args.img)
    corn = cornering(img)
    trans = persp_transf(corn)
    hi = thresh(getting_pretty(trans))
    gutSize = reSize(hi)
    bitify(gutSize)
