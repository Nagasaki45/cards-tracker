import SimpleCV

from settings import *

cam = SimpleCV.Camera(camera_index=CAMERA_INDEX)
cam.live()
