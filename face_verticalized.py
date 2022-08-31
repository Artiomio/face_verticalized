import json

import cv2
import dlib

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt




    



import math
from math import sin, cos


""" Finds the minimal horizontal rectangle containing given points
    points is a list of points e.g. [(3,5),(-1,4),(5,6)]
"""
def find_rect_range(points):
    min_x = min(points, key=lambda x: x[0])[0]
    max_x = max(points, key=lambda x: x[0])[0]

    min_y = min(points, key=lambda x: x[1])[1]
    max_y = max(points, key=lambda x: x[1])[1]
    return((min_x, min_y), (max_x, max_y))

def rotate(coords, origin, angle):
    """ Rotates given point around given origin
    """
    x, y = coords
    xc, yc = origin

    cos_angle = cos(angle)
    sin_angle = sin(angle)

    x_vector = x - xc
    y_vector = y - yc

    x_new = x_vector * cos_angle - y_vector * sin_angle + xc
    y_new = x_vector * sin_angle + y_vector * cos_angle + yc
    return (x_new, y_new)

from math import pi
def rotate_image(image, center, angle):
    row,col = image.shape[: 2]
    rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def distance(x1,y1, x2, y2):
    return math.sqrt( (x1-x2)**2 + (y1-y2)**2)


def distance_N_dim_squared(a, b):
    return sum((np.array(a) - np.array(b)) ** 2)

def magic_distances_from_landmarks(a):


    # Первая опорная точка
    xc1 = a[27][0]
    yc1 = a[27][1]


    # Вторая опорная точка
    xc2 = a[57][0]
    yc2 = a[57][1]


    distances = []

    for (x, y) in a:
        """
        x.append(i[0])
        y.append(i[1])
        """
        

        d1 = distance(x,y, xc1, yc1) 
        d2 = distance(x,y, xc2, yc2) 
        
        distances.append(d1)
        distances.append(d2)
    return distances



def normalized_landmark_vector(landmarks):
    """ Нормализация "волшебных" точек
        На данный момент:
            Вертикальная ориентация лица
            Приведение к единичному масштабу
    """

    # Считаем угол таким образом, что положительное направление - склонённость к правому плечу
    # Центр - 28-я точка - т.е. landmarks[27]

    nose_bridge = landmarks[27]

    eyes_vector_x, eyes_vector_y = landmarks[45][0] - landmarks[36][0], landmarks[45][1] - landmarks[36][1]
    angle = - math.atan(eyes_vector_y / eyes_vector_x)
    
    
    #print("Угол равен %f градусов (наклон к правому плечу)" % (angle * 180 / math.pi))
    verticalized = [rotate((x,y), origin = nose_bridge, angle = angle) for (x, y) in landmarks]

    # Временно - как хеш лица используем только глаза
    # verticalized = verticalized[42:48] + verticalized[36:42]

    ((x1, y1), (x2, y2)) = find_rect_range(verticalized)
    width = x2 - x1
    height = y2 - y1
    
 
    normalized = verticalized
    normalized = [((x-x1) / width, (y-y1) / width) for (x, y) in verticalized]
    return normalized

def magic_distances_from_image(img):
    faces, confidence, idx = detector.run(img, 1) # Запускаем поиск лиц
    face = faces[0]
    if len(faces) < 1:
        raise Exception
        
    shape = predictor(img, face) # Getting landmarks (magic dots)
    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]
    distances = magic_distances_from_landmarks(normalized_landmark_vector([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]))
    return distances
    

import dlib
predictor_path = r"/home/art/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)



cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()
    

    faces, confidence, idx = detector.run(img, 1) # Запускаем поиск лиц  
    
    if len(faces) < 1:
            cv2.putText(img, "Can't see your face", (200, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        
        face = faces[0]
        shape = predictor(img, face) 
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]
        for x, y in landmarks:
            cv2.circle(img, (x, y), 2, (50, 50, 50), -1)

#        distances = magic_distances_from_landmarks(normalized_landmark_vector([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]))

        
        #emotion_name = sorted([(distance_N_dim_squared(vector, distances), emotion_name) for vector, emotion_name in emotions_vectors])[0][1]
        #cv2.putText(img, emotion_name, (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 1, cv2.LINE_AA)
        
        #cv2.putText(img, str([(distance_N_dim_squared(vector, distances), emotion_name) for vector, emotion_name in emotions_vectors][0]), (10, 390), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(img, str([(distance_N_dim_squared(vector, distances), emotion_name) for vector, emotion_name in emotions_vectors][1]), (10, 420), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        
        
        


        try:
            pass
            #cv2.putText(img, f" {round(100 * gender_score)}", (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, round(gender_score * 255)), 1, cv2.LINE_AA)
        except Exception as e:
            print("Error!", e)
    



    nose_bridge = landmarks[27]                                                
    eyes_vector_x, eyes_vector_y = landmarks[45][0] - landmarks[36][0], landmarks[45][1] - landmarks[36][1]
    angle = math.atan(eyes_vector_y / eyes_vector_x)                           
                                                                               
    vert_img = rotate_image(img, center=nose_bridge, angle=180 * angle / pi)
    cv2.imshow('verticalized', np.concatenate([vert_img, img], axis=1))
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()
Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))