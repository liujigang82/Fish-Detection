'''
import numpy as np
import cv2 as cv
cap = cv.VideoCapture('test.mp4')
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    cv.imshow("original", frame)
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
'''
import numpy as np
import cv2
KERN_SIZE = 8
kernlen = KERN_SIZE
kern = np.ones((kernlen,kernlen))/(kernlen**2)
ddepth = -1
thresh_at = 100

def blur(image):
    return cv2.filter2D(image,ddepth,kern)


cap = cv2.VideoCapture('videos/test.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
while(1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    fgmask = fgbg.apply(gray)
    mask = blur(fgmask)
    ret2, mask = cv2.threshold(mask, thresh_at, 255, cv2.THRESH_BINARY)
    res = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cv2.imshow("original", frame)
    cv2.imshow('frame',fgmask)
    cv2.imshow("binary", mask)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()