import numpy as np
import cv2
import argparse

'''
This program is used to calibrate a camera for opencv
Daniel Nugent 11/6/2014
Image can be found at:
http://wiki.ros.org/camera_calibration/Tutorials/StereoCalibration?action=AttachFile&do=get&target=check-108.pdf
Otherwise any 9x7 chessboard grid will work
Instructions:
Line up camera with image
Press 'c' key to capture image you want to process
Change camera viewing angle and repeat until complete
    -It takes 14 GOOD images
Calibration coefficients are printed in terminal
Tips for different veiwing angle:
-Put target in image center
-Put target in corners
-Put target along edges
-Do extreme viewing angle on X then Y axis then X and Y axis
-Have image target take up whole image
-Rotate camera 45 degrees and 90 degrees
-Try to get unique viewing angles for every image
Tips for a GOOD Image:
***The entire chessboard must be in frame inorder to process
***Dont be jerking the camera when you capture(prevent motion blur)
***Make sure camera is focused
Example coefficients for Logitech c920 webcam:
RMS: 0.144252280465
camera matrix:
[[ 614.01269552    0.          315.00073982]
 [   0.          614.43556296  237.14926858]
 [   0.            0.            1.        ]]
distortion coefficients:  [ 0.12269303 -0.26618881  0.00129035  0.00081791  0.17005303]
'''

cnt = 0

if __name__ == '__main__':



    #parse arguments
    parser = argparse.ArgumentParser(description="Calibrate a camera for use with openCV")
    parser.add_argument('-c','--camera', default=0, action="store",
                        help='Camera source')
    parser.add_argument('-s','--samples', default=20, action="store", type = int,
                        help='Number of samples to collect')
    args, unknown = parser.parse_known_args()




    #open video capture
    #increment 0 to 1 or 2 if the wrong camera is used
    cap = cv2.VideoCapture(args.camera)


    #number of good images before processing calibration
    goodImages = args.samples

    # number of INTERNAL corners (8x6) for a (9x7) grid
    pattern_size = (8, 6)

    square_size = 1.0

    #sizing arrays
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0

    print("Please capture {0} good images".format(goodImages))
    x = 0
    while x < goodImages:


        #get image
        ret, img = cap.read();


        if img is None:
          print("Failed to read", x)
          break

        #make it black and white
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        cv2.imshow('raw',img)
        k = cv2.waitKey(0)  & 0xFF

        #process frame when user press 'c' key
        if k == ord('c'):

            print('processing {0}...'.format(x))

            h, w = img.shape[:2]
            found, corners = cv2.findChessboardCorners(img, pattern_size)
            if found:

                #refine corners
                term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
                cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

                #display processed image
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                vis = cv2.resize(vis,(0,0),fx=0.3,fy=0.3)
                cv2.imshow('vis',vis)

                #increment valid image count
                x+=1

            if not found:
                print('chessboard not found')
                continue

            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)

            print('ok')

        if k == 0x27: #esc
            break

    print("Analyzing {0} images please wait...".format(x))
    cv2.destroyAllWindows()
    #analyze images to calculte distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("RMS:\n", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients:\n", dist_coefs)
    cv2.destroyAllWindows()