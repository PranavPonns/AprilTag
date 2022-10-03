from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import time

from dt_apriltags import Detector
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

import math


camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
camera.color_effects = (128,128)
rawCapture = PiRGBArray(camera, size=camera.resolution)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    
    vector = np.vectorize(np.int_)
    a = np.array(corner)
    b = a.astype(int)
    
    corner = tuple((b[0],b[0]))
    

    imgpts = imgpts.astype(int)
    
    
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


at_detector = Detector(families='tag36h11',
                       nthreads=4,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=1,
                       debug=0)

with open('/home/pi/Desktop/AprilTag Files/test_info.yaml', 'r') as stream:
    parameters = yaml.load(stream, Loader=SafeLoader)

cameraMatrix = np.float32(parameters['sample_test']['K']).reshape((3,3))
camera_params = (cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2])
cameraDist = np.float32([-0.35184755, 0.17596916, -0.00249437, -0.00123845, -0.06169106])


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        
    # camera.resolution = (320, 240)
    # camera.framerate = 24
    # time.sleep(2)
    img = np.empty((240, 320, 1), dtype=np.uint8)
    img = np.asarray(image1, dtype=np.uint8)
    # camera.capture(image, 'bgr')
    # image = image.reshape((240, 320, 3))

    

    tags = at_detector.detect(img, True, camera_params, parameters['sample_test']['tag_size'])
    print(tags)

    # corners2 = cv2.cornerSubPix(img, tag.corners,(11,11),(-1,-1), criteria)
    # imgpts, jac = cv.projectPoints(axis, tag.pose_R, tag.pose_t, cameraMatrix, cameraDist)
    

    # image = draw(iamge, corners2,imgpts)

    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(image, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

        cv2.putText(image, str(tag.tag_id),
                org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0))
        
        corners3 = np.copy(tag.corners)
        corners3 = corners3.astype("float32")
        print(corners3.dtype)

        corners2 = cv2.cornerSubPix(img, corners3, (11,11),(-1,-1), criteria)
        
        #ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, cameraMatrix, cameraDist)
        
        imgpts, jac = cv2.projectPoints(axis, tag.pose_R, tag.pose_t, cameraMatrix, cameraDist)
        #imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cameraMatrix, cameraDist)

        image = draw(image, corners2, imgpts)

        print("Euler Angles:",rotationMatrixToEulerAngles(tag.pose_R))
    
    cv2.imshow("", image)
    
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    key = cv2.waitKey(1)

camera.close()
cv2.destroyAllWindows()