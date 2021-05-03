import cv2 as cv
import numpy as np
import warnings
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

warnings.filterwarnings("ignore")


"""Segment DIH Cells"""
# return raw image file, cell centres (x, y)


def new_segm(imgpath, bins=256, thresh=180):
    img_raw = cv.imread(imgpath, 0)
    planes = {}
    for k in range(5, 8):
        plane = np.full((img_raw.shape[0], img_raw.shape[1]), 2 ** k, np.uint8)
        res = cv.bitwise_and(plane, img_raw)
        x = res * 255
        v = np.median(x)
        if v > 0:
            x = cv.bitwise_not(x)
            v = np.median(x)
            x[x == v] = 0
        planes[str(k)] = x

    #sq_ker = np.ones((3,3),np.uint8)
    ell_ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    #rect_ker = cv.getStructuringElement(cv.MORPH_RECT,(3,3))

    planes['5'] = cv.erode(planes['5'], ell_ker, iterations=1)
    planes['5'] = cv.dilate(planes['5'], ell_ker, iterations=2)
 
    planes['765'] = planes['7'] + planes['6'] + planes['5']

    erosion1 = cv.erode(planes['765'], ell_ker, iterations=1)
    erosion1 = cv.dilate(erosion1, ell_ker, iterations=1)
    erosion1 = cv.erode(erosion1, ell_ker, iterations=1)
    erosion1 = cv.dilate(erosion1, ell_ker, iterations=1)

    #cv.namedWindow( 'ero'+str('765') , cv.WINDOW_NORMAL )
    #cv.imshow( 'ero'+str('765') , erosion1 )

    ############################################################

    #img = planes['765'].copy()
    img = erosion1.copy()
    r, c = erosion1.shape

    # global thresholding
    _, th1 = cv.threshold(erosion1, thresh, 255, cv.THRESH_BINARY)
    #cv.namedWindow( "global thresh", cv.WINDOW_NORMAL )
    #cv.imshow("global thresh",th1)

    #img = img_raw.copy()

    contours, _ = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #cimg =  cv.cvtColor(th1, cv.COLOR_GRAY2RGB)
    #cv.drawContours(cimg, contours, -1, (0,255,0), 1)

    #img_clr = cv.cvtColor(img_raw, cv.COLOR_GRAY2RGB)

    c_areas, cx, cy, diams = [], [], [], []
    list_cnt = 0

    for i in range(len(contours)):
        cnt = contours[i]
        if cnt.shape[0] > 5:
            M = cv.moments(cnt)
            if M['m00'] != 0:

                cx.append(int(M['m10']/M['m00']))
                cy.append(int(M['m01']/M['m00']))

                area = cv.contourArea(cnt)
                c_areas.append(area)
                equi_diameter = np.sqrt(4*area/np.pi)
                diams.append(equi_diameter)
                list_cnt += 1

    c_areas = np.array(c_areas)
    cx = np.array(cx)
    cy = np.array(cy)
    diams = np.array(diams)

    return img_raw, cx, cy



def cropper(img, X, Y, csize=66):
    #I=img.copy()
    img_copy = img.copy()
    r, c = img.shape
    X, Y = np.int32(X), np.int32(Y)
    XY = list(zip(X, Y))
    #I = cv.cvtColor(I, cv.COLOR_GRAY2RGB)

    cell_array = []
    cell_id = []

    for i in XY:
        x, y = i[0], i[1]
        #uncomment to mark all
        #stpt  = ( max(0,x-33), max(0,y-33) )
        #endpt = ( min(c,x+33), min(r,y+33) )

        #following codes removes cut out cells on the edges
        if x-33 >= 0 and y-33 >= 0 and x+33 <= c and y+33 <= r:
            stpt = (x-33, y-33)
            endpt = (x+33, y+33)
            indx = str(x)+"#"+str(y)
            cell_id.append(indx)
            crop_cell = img_copy[stpt[1]:endpt[1], stpt[0]:endpt[0]].copy()
            cell_array.append(crop_cell)
    return cell_id, cell_array


def segmenter(imgpath):
    print("Segmenting Cells...")
    img, X, Y = new_segm(str(imgpath), bins=256, thresh=180)
    cell_ids, cell_array = cropper(img, X, Y, csize=66)
    print("Segmentation Complete...")
    return cell_ids, cell_array