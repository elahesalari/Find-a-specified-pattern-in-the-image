import cv2
import numpy as np


def match_a(src,image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_a = cv2.imread(src)
    a = cv2.cvtColor(template_a, cv2.COLOR_BGR2GRAY)

    h,w = a.shape

    res = cv2.matchTemplate(img_gray, a, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    location = np.where(res >= threshold)


    for top_left in zip(*location[::-1]):

        image[top_left[1]:top_left[1] + h, top_left[0]: top_left[0] + w] = 255 - template_a
        cv2.rectangle(image, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 1)

    return image



if __name__=='__main__':
    img_text = 'images\\text.png'
    src = 'images\\a.PNG'
    src2 = 'images\\a_rot.PNG'
    main_img = cv2.imread(img_text)
    image = match_a(src, main_img)
    image = match_a(src2, image)

    cv2.imshow('resualt', image )
    cv2.waitKey(0)