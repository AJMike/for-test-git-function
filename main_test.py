# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from main import *
import os



cam_k1 = np.array([[902.5117224246152, 0.0, 609.3680245880764], [0.0, 900.8669374105964, 331.33559398971926], [0.0, 0.0, 1.0]])
cam_d1 = np.array([-0.4027689751429306, 0.12573566945051362, 0.00040387440037552405, -0.0005013980352025684, 0.0])

def Undistort(img_data,cam_k,cam_d):

    h, w = img_data.shape[:2]

    mapx, mapy = cv2.initUndistortRectifyMap(cam_k, cam_d, None, cam_k, (w, h), 5)

    image_undistort = cv2.remap(img_data, mapx, mapy, cv2.INTER_LINEAR)

    return image_undistort

path = '/media/yxy/FSK-Standard-1/bad_rr/6ms/left/'
file_list = os.listdir(path)
file_list.sort()
for i in file_list:
	img = cv2.imread(path+i)
	img_ready = Undistort(img,cam_k1,cam_d1)
	info = scnn(img_ready)
	print(info)
