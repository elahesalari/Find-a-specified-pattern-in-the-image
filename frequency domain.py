import numpy as np
import scipy.fftpack as fp
from skimage.io import imread
from skimage.color import rgb2gray, gray2rgb
from skimage.draw import rectangle_perimeter
import cv2


def matching( image, temp):
    rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    template = rgb2gray(imread(temp))

    h, w = template.shape

    location = find_loc(image, template)
    location_rot = find_loc(rotate, template)

    # print(location)
    # print(location_rot)

    # draw rectangle over the rotate image
    final_rot = (gray2rgb(rotate)).astype(np.uint8)

    for top_left in location_rot:
        rr, cc = rectangle_perimeter(top_left, end=(top_left[0] + h, top_left[1] + w), shape=rotate.shape)
        for x in range(-1, 1):
            for y in range(-1, 1):
                final_rot[rr + x, cc + y] = (0, 255, 0)

    rotate_back = cv2.rotate(final_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # draw rectangle over the main image
    for top_left in location:
        rr, cc = rectangle_perimeter(top_left, end=(top_left[0] + h, top_left[1] + w), shape=image.shape)
        for x in range(-1, 1):
            for y in range(-1, 1):
                rotate_back[rr + x, cc + y] = (0, 255, 0)

    return rotate_back


def find_loc(image, template):
    fourier_main = fp.fftn(image)
    fourier_a = fp.fftn(template, image.shape)
    f_conj = fourier_main * np.conj(fourier_a)
    f_back = (fp.ifftn(f_conj / np.linalg.norm(f_conj))).real

    # find best template
    cross_re = f_back.reshape(image.shape[0] * image.shape[1])
    argmax_c = cross_re.argmax()

    idx_a = []
    threshold = 0.00001
    for i in range(len(cross_re)):
        if abs(cross_re[i] - cross_re[argmax_c]) < threshold :
            idx_a.append(i)
    location = []
    for p in idx_a:
        top_left = np.unravel_index(p, f_back.shape)
        location.append(top_left)

    return location


if __name__ == '__main__':
    src = 'images\\a.PNG'
    main = rgb2gray(imread('images\\text.png'))
    img = matching( main, src)
    cv2.imshow('resualt', img)
    cv2.waitKey(0)
