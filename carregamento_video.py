import cv2
import numpy
from file_utils import existeArquivo

def carrega_arquivo_video(caminho:str):
    existeArquivo(caminho)
    video = cv2.VideoCapture(caminho)
    numero_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    largura = numpy.int32(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = numpy.int32(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = carrega_frames_video(video)
    video.release()

    return numero_frames, frames, fps, largura, altura

def carrega_frames_video(video):
    frames = []
    while True:
        success, frame = video.read()
        if success:
            frames.append(frame)
        else:
            break
    return frames
