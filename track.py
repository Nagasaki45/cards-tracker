'''
Tracking cards from different colors with SimpleCV.

Usage:
    python track.py [img|segmented]

Options:
    img            shows the image itself
    segmented      shows binarized image, usefull for setting
                   COLOR_BINARIZATION_THRESHOLD in the settings file

Based on the code from:
http://www.youtube.com/watch?v=jihxqg3kr-g
'''

import time
import sys
import numpy as np
import SimpleCV

from settings import *


def dist_from_color(img, color):
    '''
    SimpleCV.Image, tuple -> int

    tuple: (r, g, b)
    '''

    # BUG in getNumpy, it returns with colors reversed
    matrix = (img.getNumpy()[:, :, [2, 1, 0]] - color) ** 2
    width, height = img.size()
    return matrix.sum() ** 0.5 / (width * height)


def main():

    print(__doc__)
    cam = SimpleCV.Camera(camera_index=CAMERA_INDEX)
    display = False
    if len(sys.argv) > 1:
        display = sys.argv[1]

    # wait some time for the camera to turn on
    time.sleep(1)
    background = cam.getImage()

    print('Everything is ready. Starting to track!')

    while True:

        img = cam.getImage()
        dist = ((img - background) + (background - img)).dilate(5)
        # segmented = dist
        segmented = dist.binarize(COLOR_BINARIZATION_THRESHOLD).invert()
        blobs = segmented.findBlobs(minsize=CARD_DIMENSION ** 2)
        if blobs:
            points = []
            for b in blobs:
                points.append((b.x, b.y))
                car = img.crop(b.x, b.y,
                               CARD_DIMENSION, CARD_DIMENSION, centered=True)
                # color distances from cars
                dists = [dist_from_color(car, c['color']) for c in CARDS]
                choosen_car = CARDS[np.argmin(dists)]['name']
                print(b.x, b.y, choosen_car)

        if display:
            to_show = locals()[display]
            if blobs:
                to_show.drawPoints(points)
            to_show.show()
        else:
            time.sleep(0.1)


if __name__ == '__main__':
    main()
