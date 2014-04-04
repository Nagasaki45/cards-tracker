'''
Tracking cards from different colors with SimpleCV.

Based on the code from:
http://www.youtube.com/watch?v=jihxqg3kr-g
'''

import time
import numpy as np
import matplotlib.pyplot as plt
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
    display = SimpleCV.Display()
    cam = SimpleCV.Camera(camera_index=CAMERA_INDEX)
    normaldisplay = True

    # wait some time for the camera to turn on
    time.sleep(1)
    background = cam.getImage()

    while display.isNotDone():

        if display.mouseRight:
            normaldisplay = not(normaldisplay)
            print('Display Mode: {}'.format(
                'Normal' if normaldisplay else 'Segmented'))

        img = cam.getImage()
        dist = ((img - background) + (background - img)).dilate(5)
        # segmented = dist
        segmented = dist.binarize(COLOR_BINARIZATION_THRESHOLD).invert()
        # find blobs defaults
        # threshval=-1, minsize=10, maxsize=0, threshblocksize=0,
        # threshconstant=5, appx_level=3
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
            img.drawPoints(points)

        if normaldisplay:
            img.show()
        else:
            segmented.show()


if __name__ == '__main__':
    main()
