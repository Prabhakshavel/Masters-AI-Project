# import the necessary packages
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import dlib
import cv2
import extrautils


def plt_imshow(title, image):
    # convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()

def featureProcess(shape_predictorEnt, imageEnt):

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictorEnt)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(imageEnt)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # initialize unique color for each facial landmark region
    colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
              (168, 100, 168), (158, 163, 32),
              (163, 38, 32), (180, 42, 220),
              (100, 68, 109)]

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        output2 = extrautils.handle_facial_landmarks(image, shape, imagepath=imageEnt)
        #plt_imshow("Image", output)

def main(shape_predictor, image):
    featureProcess(shape_predictor, image)

if __name__ == "__main__":
    main()
