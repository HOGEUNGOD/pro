from numpy.lib.stride_tricks import as_strided
import os
import cv2
import sys
from scipy import io
import pandas as pd
from scipy.spatial import distance
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import mecha.fracture

def pool2d(A, kernel_size, stride, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    if pool_mode == 'min':
        value = A.max()
    else:
        value = 0
    # Padding
    A = np.pad(A, kernel_size-2 , mode='constant', constant_values=value)

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'min':
        return A_w.min(axis=(1,2)).reshape(output_shape)

def make_mask(path):
    class PolygonDrawer(object):
        def __init__(self, window_name):
            self.window_name = window_name # Name for our window
            self.done = False # Flag signalling we're done
            self.current = (0, 0) # Current position, so we can draw the line-in-progress
            self.points = [] # List of points defining our polygon

        def on_mouse(self, event, x, y, buttons, user_param):
            # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
            if self.done: # Nothing more to do
                return
            if event == cv2.EVENT_MOUSEMOVE:
                # We want to be able to draw the line-in-progress, so update current mouse position
                self.current = (x, y)
            elif event == cv2.EVENT_LBUTTONDOWN:
                # Left click means adding a point at current position to the list of points
                # print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
                self.points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click means we're done
                print("Completing polygon with %d points." % len(self.points))
                self.done = True

        def run(self):
            # Let's create our working window and set a mouse callback to handle events
            cv2.namedWindow(self.window_name)
            cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
            cv2.waitKey(1)
            cv2.setMouseCallback(self.window_name, self.on_mouse)

            while(not self.done):
                # This is our drawing loop, we just continuously draw new images
                # and show them in the named window
                canvas = images[i]
                if (len(self.points) > 0):
                    # Draw all the current polygon segments
                    cv2.polylines(canvas, np.array([self.points]), False, (255, 255, 255), 1)
                    # And  also show what the current segment would look like
                    # cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
                # Update the window
                cv2.imshow(self.window_name, canvas)
                # And wait 50ms before next iteration (this will pump window messages meanwhile)
                if cv2.waitKey(50) == 27: # ESC hit
                    self.done = True
            # User finised entering the polygon points, so let's make the final drawing
            canvas = images[i]
            # of a filled polygon
            if (len(self.points) > 0):
                cv2.fillPoly(canvas, np.array([self.points]), (255, 255, 255))
                # print(np.array([self.points]))
            # And show it
            cv2.imshow(self.window_name, canvas)
            # Waiting for the user to press any key
            cv2.waitKey()
            cv2.destroyWindow(self.window_name)
            return canvas, self.points

    # file_list = os.listdir(path)
    # img = [file for file in file_list if file.endswith(".jpg")]
    # image = cv2.imread(path+'/'+img[num], cv2.IMREAD_GRAYSCALE)
    images = load_file(path,flip='')

    # if image is None:
    #     print('Image load failed!')
    #     sys.exit()
    for i in images:
        x, y, w, h = cv2.selectROI(images[i])
        cv2.destroyAllWindows()

        cv2.rectangle(images[i], (x,y), (x+w,y+h),(0,0,0),2)
        CANVAS_SIZE = np.shape(images[i])

        poly = PolygonDrawer("Polygon")
        _, poly_points = poly.run()

        ##mask part
        mask = np.zeros(np.shape(images[i]))
        mask[y:y+h, x:x+w] = 255
        cv2.fillPoly(mask, np.array([poly_points]), (0,0,0))
        cv2.imshow('mask', mask)

        #%% save part
        cv2.imwrite(path+'/'+str(i)+'_mask.png',mask)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return

class Crack_tip():
    def __init__(self, ratio, key, initial_crack, thickness, width):
        """ x, y 는 좌표, point 는 리스트로 x,y좌표 모아줌, count는 좌표 몇개, 길이, 1픽셀당 mm(mm/1픽셀)"""
        self.B = thickness
        self.W = width
        self.x = 0
        self.y =0
        self.inital_crack = initial_crack
        self.point = []
        self.count= -1
        self.length = 0
        self.ratio = ratio
        self.df = pd.DataFrame(index = [i for i in key], columns=['length(mm)',  'length(add)(mm)', 'x', 'y', 'Load(N)','ASTM_E399(K)'])

        self.key= list(key)
        return

    def __inital__(self, x, y, load):
        self.x = x
        self.y= y
        self.point.append([x,y])
        self.df.loc[self.key[0]] = [self.inital_crack ,self.inital_crack , x, y,
                                    load, ASTM_E399(load*0.001,self.inital_crack*0.1,self.B*0.1,self.W*0.1)]

        self.count +=1


    def append(self, x, y, load):
        if self.count == -1:
            self.__inital__(x,y,load)
        else:
            self.x =x
            self.y= y
            self.point.append([x,y])
            self.count +=1

            self.length = distance.cdist([self.point[self.count-1]], [self.point[self.count]]) * self.ratio
            self.df.loc[self.key[self.count]] = [ self.length[0][0] ,self.length[0][0] + self.df["length(add)(mm)"][self.count-1], x, y,
                                                  load, ASTM_E399(load*0.001, (self.length + self.df["length(add)(mm)"][self.count-1])[0][0]*0.1,self.B*0.1,self.W*0.1)]
            print(type((self.length + self.df["length(add)(mm)"][self.count-1])[0][0]))
    def position(self):
        return self.point

    def all_position(self):
        return self.df
    pass


def ASTM_E399(Pq,a,B,W):
    """B is thickness(cm), W is width(cm), Pq is stress(kN), A is crack length(cm)"""
    f_in = a/W
    f=((2+f_in)*(0.886+4.64*f_in-13.32*(f_in**2)+14.72*(f_in**3)-5.6*(f_in**4)))/((1-f_in)**1.5)
    kq=(Pq*f)/(B*(W**0.5))
    return kq

def hookes_law_sigmayy(YoungsModulus, Poisson, strain_xx, strain_yy):
    sigma_yy = YoungsModulus*(strain_xx*Poisson + strain_yy)/(1-Poisson**2)
    return sigma_yy

def Williams_sigmayy(r, theta):
    theta = math.radians(theta)
    K1_coefficient = np.cos(theta/2) * (1 + np.sin(theta/2) * np.sin(3 * theta / 2)) / (2 * np.pi * r) ** 0.5
    K2_coefficient = np.sin(theta/2) * np.cos(theta/2) * np.cos(3*theta/2) / (2 * np.pi * r) ** 0.5
    return [K1_coefficient, K2_coefficient]

def cart2pol(x, y):
    """out put : dgree, rho"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    degree = np.rad2deg(phi).reshape([-1,1])
    out = np. hstack((rho,degree))
    return out

def pol2cart(rho, degree):
    """in put : degree, rho"""
    phi = np.deg2rad(degree).reshape([-1,1])
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    out = np.hstack((x, y))
    return out

def circle_location(radius, ratio, degree, rotate_degree=0):
    """
    첫번째 원점에서 떨어져있는 값 radius 반지름길이임 (픽셀단위)
    예상각도 수정하고 싶으면 np.arrange 안에 각도 바꿔주면됨
    ratio : mm/pixel , output mm
    x , y 좌표 , rho, theta(degree)
   """
    degree = np.arange(-1*degree+rotate_degree, rotate_degree+degree+1 ,1)
    degree = degree.reshape([-1,1])
    rho = np.ones([degree.shape[0],1]) * radius
    xy = pol2cart(rho, degree).round()
    rtheta = np.hstack((degree, rho * ratio))
    out = np.hstack((xy, rtheta))
    return out

def find_zone(array):
    """
    :param array: 일반 배열받음
    :return: 주위로 둘러쌓인 0값을 제거하고 행렬 반환
    """
    y_axis = np.nonzero(array)[0]
    x_axis = np.nonzero(array)[1]
    y_min, y_max = np.min(y_axis), np.max(y_axis)
    x_min, x_max = np.min(x_axis), np.max(x_axis)
    result = array[y_min:y_max, x_min:x_max]
    return result

def u_f1(n, r, theta):
    shear_modulus = 26900 #unit:mpa
    poisson_ratio = 0.33
    theta = np.deg2rad(theta)
    plane_stress= (3 - poisson_ratio)/(1 + poisson_ratio)
    plane_strain = 3 - 4 * poisson_ratio

    result = r**(n/2) * (plane_stress + n/2 + (-1)**n * np.cos(n/2 * theta) - n/2*np.cos((n/2 - 2) * theta)) / (2 * shear_modulus)
    return result

def u_f2(n, r, theta):
    shear_modulus = 26900 #unit:mpa
    poisson_ratio = 0.33
    theta = np.deg2rad(theta)
    plane_stress= (3 - poisson_ratio)/(1 + poisson_ratio)
    plane_strain = 3 - 4 * poisson_ratio

    result = r**(n/2) * (plane_stress + n/2 -(-1)**n * np.sin(n/2*theta) - 0.5*n*np.sin((n/2-2)*theta)) / (2 * shear_modulus)
    return result

def v_f1(n, r, theta):
    shear_modulus = 26900 #unit:mpa
    poisson_ratio = 0.33
    theta = np.deg2rad(theta)
    plane_stress= (3 - poisson_ratio)/(1 + poisson_ratio)
    plane_strain = 3 - 4 * poisson_ratio

    result = r**(n/2) * (plane_stress - n/2 -(-1)**n * np.sin(n/2 * theta) + n/2*np.sin((n/2 -2)*theta)) / (2 * shear_modulus)
    return result

def v_f2(n, r, theta):
    shear_modulus = 26900 #unit:mpa
    poisson_ratio = 0.33
    theta = np.deg2rad(theta)
    plane_stress= (3 - poisson_ratio)/(1 + poisson_ratio)
    plane_strain = 3 - 4 * poisson_ratio

    result = r**(n/2) * (-1 * plane_stress + n/2 -(-1)**n * np.cos(n/2 *theta) - 0.5*n*np.cos((n/2 - 2)  * theta)) / (2 * shear_modulus)
    return result

def load_file(path, flip = 'None'):
    """ 폴더에 들어간 순서대로 이미지 추가 Dict형태 좌우 반전시 flip = 'flip' 사용"""
    files = glob.glob(path+'./*.jpg')
    if not files:
        print("Chck Phath")
        sys.exit()
    img = {}
    for i in files:
        im = cv2.imread(i)
        name = os.path.split(os.path.splitext(i)[0])[-1]
        if flip == 'flip':
            im = cv2.flip(im, 1)
        img.setdefault('%s'%name,im)
    return img