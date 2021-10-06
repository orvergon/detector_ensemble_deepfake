import face_recognition.api
from PIL import Image, ImageDraw
import cv2
from otimizacao_frame import grayscale
import image_utils
import numpy

downsize= 0.7


def extrair_pontos_face(frame):
    raw_landmarks = face_recognition.api._raw_face_landmarks(frame)
    faces = list()
    for face in raw_landmarks:
        landmarks_x_pos = [int(aux.x / 1) for aux in face.parts()]
        landmarks_y_pos = [int(aux.y / 1) for aux in face.parts()]
        final_landmarks = list(zip(landmarks_x_pos, landmarks_y_pos))
        faces.append(final_landmarks)
    return faces

def extrair_pontos_face_multi_frames(frame_list:list):
    face_list = [[], [], [], []]  #Numero máximo de faces reconheciveis em um frame, definido como 4
    count = 0
    for frame in frame_list:
        faces = extrair_pontos_face(frame)
        count += 1
        for index, face_points in enumerate(faces):
            try:
                face_list[index].append(face_points)
            except IndexError:
                continue
    return face_list