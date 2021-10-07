# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import imutils
import re

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# default the indexes to the
# 68-point model
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def handle_facial_landmarks(image, shape, imagepath=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if imagepath is None:
        imagepath = "error.png"

    # decide what image should be named
    imageID = str(imagepath)
    txt = imageID
    #imageID = re.search("[ABLW][MF]-[0-2][0-9][0-9]", txt)
    imageID = re.search("[ABLW]-[0-2]", txt)
    #txt = imageID.group(0)
    txt = imagepath[-6:-4]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        if name == "mouth":
            # rect = cv2.boundingRect(pts)
            # x,y,w,h = rect
            # crop image
            #roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            (x, y, w, h) = cv2.boundingRect(np.array(pts))
            # initialise max values for consistent size
            mx = 375
            my = 190
            wa = int((mx - w)/2)
            ha = int((my-h)/2)
            roi = image[y:y + h, x:x + w]
            #roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            # make mask
            pts = pts - pts.min(axis=0)

            mask = np.zeros(roi.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
            # channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            # ignore_mask_color = (255,) * channel_count
            # cv2.fillPoly(mask, np.int32([pts]), ignore_mask_color)
            # bitwise and
            dst = cv2.bitwise_and(roi, roi, mask=mask)

            # background
            bg = np.ones_like(roi, np.uint8) * 255
            cv2.bitwise_not(bg, bg, mask=mask)
            dst2 = bg + dst
            dst3 = cv2.copyMakeBorder(dst2, ha, ha, wa, wa, cv2.BORDER_CONSTANT, value=(255,255,255))
            #dst4 = imutils.resize(dst3, width=375, height=190, inter=cv2.INTER_CUBIC)
            dst4 = cv2.resize(dst3, (375, 190))
            # output function
            outputPath = "D:/Dissertation/Datasets/outputset/mouths/" + txt + ".png"
            cv2.imwrite(outputPath, dst4,[cv2.IMWRITE_PNG_COMPRESSION,0])

        if name == "right_eyebrow":
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            # crop image
            # initialise max values for consistent size
            mx = 250
            my = 150
            wa = int((mx - w) / 2)
            ha = int((my - h) / 2)
            roi = image[y - ha:y + h + ha, x - wa:x + w + wa]
            #roi = image[(y - 15):y + h + 10, (x - 15):x + w + 15]
            #roi = imutils.resize(roi, width=250, height=125, inter=cv2.INTER_CUBIC)
            roi = cv2.resize(roi, (250, 125))

            # output function
            outputPath = "D:/Dissertation/Datasets/outputset/eyebrows/" + txt + "-1.png"
            cv2.imwrite(outputPath, roi)

        if name == "left_eyebrow":
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            # crop image
            # initialise max values for consistent size
            mx = 250
            my = 150
            wa = int((mx - w) / 2)
            ha = int((my - h) / 2)
            roi = image[y - ha:y + h + ha, x - wa:x + w + wa]
            #roi = image[(y - 15):y + h + 10, (x - 15):x + w + 15]
            #roi = imutils.resize(roi, width=250, height=125, inter=cv2.INTER_CUBIC)
            roi = cv2.resize(roi, (250,125))

            # mirror image
            roi = np.fliplr(roi)

            # output function
            outputPath = "D:/Dissertation/Datasets/outputset/eyebrows/" + txt + "-2.png"
            cv2.imwrite(outputPath, roi)

        if name == "right_eye":
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            # crop image
            # initialise max values for consistent size
            mx = 250
            my = 150
            wa = int((mx - w) / 2)
            ha = int((my - h) / 2)
            roi = image[y - ha:y + h + ha, x - wa:x + w + wa]
            #roi = image[(y - 25):y + h + 15, (x - 20):x + w + 20]
            #roi = imutils.resize(roi, width=250, height=150, inter=cv2.INTER_CUBIC)
            roi = cv2.resize(roi, (250, 150))

            # output function
            outputPath = "D:/Dissertation/Datasets/outputset/eyes/" + txt + "-1.png"
            cv2.imwrite(outputPath, roi)

        if name == "left_eye":
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            # crop image
            # initialise max values for consistent size
            mx = 250
            my = 150
            wa = int((mx - w) / 2)
            ha = int((my - h) / 2)
            roi = image[y - ha:y + h + ha, x - wa:x + w + wa]
            #roi = image[(y - 25):y + h + 15, (x - 20):x + w + 20]
            #roi = imutils.resize(roi, width=250, height=150, inter=cv2.INTER_CUBIC)
            roi = cv2.resize(roi, (250, 150))

            # mirror image
            roi = np.fliplr(roi)

            # output function
            outputPath = "D:/Dissertation/Datasets/outputset/eyes/" + txt + "-2.png"
            cv2.imwrite(outputPath, roi)

        if name == "nose":
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            # crop image
            # grab values for mouth and eyebrows
            (browj,browk) = FACIAL_LANDMARKS_IDXS["left_eyebrow"]
            (mouthj,mouthk) = FACIAL_LANDMARKS_IDXS["mouth"]
            browpts = shape[browj:browk]
            mouthpts = shape[mouthj:mouthk]
            (browx, browy, broww, browh) = cv2.boundingRect(np.array(browpts))
            (mouthx, mouthy, mouthw, mouthh) = cv2.boundingRect(np.array(mouthpts))
            browLow = browy+browh
            mouthHigh = mouthy
            # initialise max values for consistent size
            mx = 250
            my = mouthHigh
            wa = int((mx - w) / 2)
            ha = my - y + 25
            roi = image[browLow:browLow + ha, x - wa:x + w + wa]
            #roi = image[(y - 15):y + h + 15, (x - 15):x + w + 15]
            #roi = imutils.resize(roi, width=250, height=475, inter=cv2.INTER_CUBIC)
            roi = cv2.resize(roi, (250, 475))

            # output function
            outputPath = "D:/Dissertation/Datasets/outputset/noses/" + txt + ".png"
            cv2.imwrite(outputPath, roi)


    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    return output
