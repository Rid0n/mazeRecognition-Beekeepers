import cv2
import numpy as np
import argparse
from collections import Counter
size = (17,17) # размер лабиринта


def cornering(img):
    img = cv2.bilateralFilter(img,10,17,17)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(hsv, np.array((38,0,0)), np.array((255,255,255)))
    mask = 255 - thresh
    eroder = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, eroder, iterations=1)

    dst = cv2.cornerHarris(mask, 2, 3, 0.04)  # ищем углы по Харрису
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    mask = np.zeros_like(hsv)
    mask[dst > 0.01 * dst.max()] = 255
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

    coor = np.argwhere(mask) #

    coor = np.flip(coor)
    corners = np.zeros((4, 2), dtype="float32")

    s = np.float32(coor).sum(axis=1)
    corners[0]=coor[np.argmin(s)]
    corners[3]=coor[np.argmax(s)]           # ищем углы лабиринта
    d = np.diff(np.float32(coor),axis=1)
    corners[2]=coor[np.argmin(d)]
    corners[1]=coor[np.argmax(d)]
    # img2 = img.copy()
    # for pt in corners:
    #     cv2.circle(img2, tuple((pt)), 3, (0, 0, 255), -1)  # посмотреть, что углы верно определились
    # cv2.imshow('io',img2)
    # cv2.waitKey(10000)
    return corners

def persp_transf(corners):
    pts1 = corners
    pts2 = np.float32([[0,0],[20*size[0],0],[0,20*size[1]],[20*size[0],20*size[1]]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    result = cv2.warpPerspective(img,matrix,(20*size[0],20*size[1]))
    # приведение перспективы
    return result


#cv2.imshow('v',im_bw)
def weedOutTheGreens(img):
    filt_img = cv2.bilateralFilter(img, 10, 17, 17)
    hsv = cv2.cvtColor(filt_img, cv2.COLOR_BGR2HSV)
    sh, ss, sv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    ch, cs, cv = Counter(sh.flat), Counter(ss.flat), Counter(sv.flat)

    c = [ch,cs,cv]
    #по наиболее часто встречающемуся цвету выделяем лабиринт
    b = [None] * 3

    for j in range(3):
        for i in range(len(c[j]) - 1):
            b[j] = (c[j].most_common(i + 1))[-1][0]
            if b[j] != 0 :
                break

    res = cv2.inRange(hsv, np.array((b[0] - 5, b[1] - 5, b[2] - 5)),np.array((b[0] + 5, b[1] + 5, b[2] + 5)))

    im = np.flip(res, 1)
    im = np.rot90(im)

    return im



def reSize(im):
    im = cv2.resize(im, size, interpolation=cv2.INTER_BITS)
    return im# сводим размер к желаемому

def bitify(im):
    im = np.where(im > 150, 1, 0)
    nim = np.array(im)
    h,w=im.shape
# Print that puppy out
    for r in range(h):
        for c in range(w):
            print(nim[r,c],end='')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('image', type=str, help="Image to Bi-matrify")
    args = parser.parse_args()
    img = cv2.imread(args.image)
    corn = cornering(img)
    trans = persp_transf(corn)
    hi = weedOutTheGreens(trans)
    gutSize = reSize(hi)
    bitify(gutSize)
