import pywt
import cv2
from video_data import VideoData
import numpy
from math import sqrt
from numpy.lib.stride_tricks import as_strided
import image_utils
import blur_wavelet


treshold = 35


def get_bouding_box(landmarks:list[tuple[int, int]]):
    max_x, max_y = (0, 0)
    min_x, min_y = (10**100, 10**100)
    for x, y in landmarks:
        max_x = max(x, max_x)
        max_y = max(y, max_y)

        min_x = min(x, min_x)
        min_y = min(y, min_y)
    return ((max_x, max_y), (min_x, min_y))


def flat_pontos_faciais(pontos_faciais):
    flat = list()
    for face in pontos_faciais:
        for frame in face:
            flat.append(frame)
    return flat


def analisa_video_rodenas(video_data:VideoData):
    flat_landmarks = flat_pontos_faciais(video_data.pontos_faciais)
    blur_ext_list_face = list()
    blur_per_list_face = list()
    blur_ext_list_frame = list()
    blur_per_list_frame = list()
    for i, landmarks in enumerate(flat_landmarks):
        try:
            frame = numpy.copy(video_data.frames_originais[i])
        except:
            return 1
        bounding_box = get_bouding_box(landmarks)
        face_crop = numpy.copy(frame[bounding_box[1][1]:bounding_box[0][1], bounding_box[1][0]:bounding_box[0][0]])
        frame[bounding_box[1][1]:bounding_box[0][1], bounding_box[1][0]:bounding_box[0][0]] = 0
        blur_face = blur_wavelet.blur_detect(face_crop, treshold)
        blur_frame = blur_wavelet.blur_detect(frame, treshold)
        blur_per_list_frame.append(blur_frame[0])
        blur_ext_list_frame.append(blur_frame[1])
        blur_per_list_face.append(blur_face[0])
        blur_ext_list_face.append(blur_face[1])

    blur_frame = sum(blur_ext_list_frame)/len(blur_ext_list_frame)
    blur_face = sum(blur_ext_list_face)/len(blur_ext_list_face)
    return (blur_face/blur_frame)


def analisa_video(video_data:VideoData):
    for frame in video_data.frames_originais:
        haar1, haar2, haar3 = processa_haar_recursivo(frame, 3)
        print(haar1)


def mapa_edge(LH, HL, HH):
    return numpy.sqrt(numpy.power(LH, 2)+numpy.power(HL, 2)+numpy.power(HH, 2))


def processa_haar_recursivo(imagem, profundidade=3):
    if profundidade == 0:
        return []
    LL, (LH, HL, HH) = pywt.dwt2(imagem, "haar")
    return [mapa_edge(LH, HL, HH), *processa_haar_recursivo(LL, profundidade-1)]
