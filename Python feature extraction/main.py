import os
import re
import feature_extraction
import subprocess

rootdir = 'D:/Dissertation/Datasets/cfd/CFD Version 3.0/Images/CFD'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print(os.path.join(subdir, file))
        #print(file)
        x = re.search("-N.jpg$", file)
        if x:
            #run code for extracting features
            imageFilePath = subdir+"/"+file
            feature_extraction.main(shape_predictor="shape_predictor_68_face_landmarks.dat", image=imageFilePath)
        else:
            #do nothing
            print(file)
