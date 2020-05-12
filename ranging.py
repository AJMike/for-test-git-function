import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def ranging(path):
    img = cv2.imread(path, 0)
    # b, g, r = cv2.split(image)
    # img = cv2.merge([r, g, b])
    img = cv2.resize(img, (1024, 768))

    pts1 = np.float32([[499, 458], [302, 605], [766, 605], [575, 458]])  # ss and yxy data mark
    pts2 = np.float32([[250, 0], [250, 767], [750, 767], [750, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img2 = cv2.warpPerspective(img, matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    '''
    print(type(pts1[0]))
    p1, p2, p3, p4 = pts1
    p1 = tuple(p1)
    p2 = tuple(p2)
    p3 = tuple(p3)
    p4 = tuple(p4)

    thick = 3

    cv2.line(img, p1, p4, (0, 255, 0), thick)
    cv2.line(img, p2, p3, (0, 255, 0), thick)
    cv2.line(img, p1, p2, (255, 255, 0), thick)
    cv2.line(img, p3, p4, (255, 255, 0), thick)
    cv2.line(img, (0, 384), (1023, 384), (255, 0, 255), thick)
    cv2.line(img, (511, 0), (511, 767), (255, 0, 255), thick)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.subplot(121)
    plt.imshow(img)
    plt.title('input')
    plt.subplot(1, 2, 2)
    plt.imshow(img1)
    plt.title('output')

    plt.show()
    '''
    # img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    w = 420 / 500  # unit: cm
    y1 = 767
    y2 = 400
    # bottom line x1-left/x2-right
    p1, p2, p3, p4 = 0, 0, 0, 0
    for x1 in range(0, round(0.4 * 1023)):
        if 100 > img2[y1, x1] > 50:
            p1 = x1
    for x2 in range(1023, 512, -1):
        if 100 > img2[y1, x2] > 50:
            p2 = x2
    # top line x3-left/x4-right
    for x3 in range(0, round(0.5 * 1023)):
        if 100 > img2[y2, x3] > 50:
            p3 = x3
    for x4 in range(1023, 512, -1):
        if 100 > img2[y2, x4] > 50:
            p4 = x4
    print(p1, p2, p3, p4)
    '''
    cv2.line(img1, (p1, y1), (p3, y2), (0, 255, 0), thick)
    cv2.line(img1, (p2, y1), (p4, y2), (0, 255, 0), thick)
    cv2.line(img1, (p1, y1), (p2, y1), (255, 255, 0), thick)
    cv2.line(img1, (p3, y2), (p4, y2), (255, 255, 0), thick)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # Angle:
    xp1 = 0.5 * (p1 + p2)
    xp2 = 0.5 * (p3 + p4)
    oe = round((xp1 - xp2), 2)  # point to right is positive
    be = round(3.1 * (y1 - y2), 2)
    r_angle = math.atan(oe / be)
    angle = (math.atan(oe / be) / math.pi) * 180
    cos = math.cos(r_angle)
    # cos1 = (be/((oe**2 + be**2)**0.5))
    if oe > 0:
        course = 'AGV points to right'
        # print(course)
    else:
        course = 'AGV points to left'
        # print(course)

    # Distances:(WARNING : all unit is centimeter meter*100 or pixel)
    # Front distances:
    def distance(d):
        d_bottom = round(d / w)
        cam2right = 140
        cam2left = 140

        delta = (xp1 - xp2) / ((y1 - y2) / (y1 - y2 + d_bottom))
        if delta > 0:

            d_right = (-delta + (p4 - 511)) * w - cam2right
            d_left = (511 - p3 + delta) * w - cam2left
        else:
            d_right = (delta + (p4 - 511)) * w - cam2right
            d_left = (511 - p3 - delta) * w - cam2left

        return d_left * cos, d_right * cos

    d_l1, d_r2 = distance(290)
    # print('200cm beyond head distances:')
    # print('Left:', d_l1, 'cm' + '\nRight:', d_r2, 'cm')
    d_l3, d_r4 = distance(490)
    # print('Front distances:')
    # print('Left:', d_l3, 'cm' + '\nRight:', d_r4, 'cm')
    d_l5, d_r6 = distance(1680)
    # print('Back distances:')
    # print('Left:', d_l5, 'cm' + '\nRight:', d_r6, 'cm')

    return str(course)+' '+str(angle)+' '+str(d_l1)+' '+str(d_r2)+' '+str(d_l3)+' '+str(d_r4)+' '+str(d_l5)+' '+str(d_r6)
