from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2

camera = PiCamera(resolution=(640 , 480), framerate=30)
camer.iso = 100
sleep(2)

camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = "off"
g = camera.awb_gains
camera.awb_mode = "off"
camera.awb_gains = g



camera.color_effects = (128,128)

for x in range('img{counter:03d}.jpg'):
    print('Captured %s' % filename)
    sleep(300) # wait 5 minutes


