#!/usr/bin/python

import sys
import os
import dlib
import glob
from skimage import io
from scipy.spatial import distance

predictor_path = "../data/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "../data/dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "../examples"


if len(sys.argv) != 3:
    for s in sys.argv:
        print s
    print(
        "Call this program like this: wtf wtf\n")
    exit()

face1 = sys.argv[1]
face2 = sys.argv[2]




# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


# Now process all the images
f = face1
print("Processing file: {}".format(f))
img = io.imread(f)
print "yahan tak ho gaya hai"
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

# Now process each face we found.
for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    

f = face2
print("Processing file: {}".format(f))
img = io.imread(f)
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

# Now process each face we found.
for k, d in enumerate(dets):
    shape = sp(img, d)
    face_des = facerec.compute_face_descriptor(img, shape)


dst = distance.euclidean(face_descriptor,face_des)
print(dst)
dlib.hit_enter_to_continue()
