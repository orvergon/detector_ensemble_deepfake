import traceback
import glob
import os
import time
import json
from carregamento_video import carrega_arquivo_video
from otimizacao_frame import otimiza_frames
import extracao_caracteristicas
from posicao_facial import processa_poses_video
from video_data import VideoData
from classificador_pos_facial_svm import classifica_faces
from wavelet_transform import analisa_video_rodenas
from classificador_fft import classifica_video_fft
import pickle

classificador = pickle.load(open("svm.pkl", "rb"))

pickle_data = False

def roda_video(nome_video:str, label:str):
    video_data = VideoData()
    video_data.nome = nome_video
    video_data.label = label

    numero_frames, frames, fps, largura, altura = carrega_arquivo_video(nome_video)
    video_data.fps = fps
    video_data.frames_originais = frames[10:40]
    video_data.largura = largura
    video_data.altura = altura

    video_data.frames_otimizados = otimiza_frames(video_data.frames_originais)
    video_data.pontos_faciais = extracao_caracteristicas.extrair_pontos_face_multi_frames(video_data.frames_otimizados)
    video_data.pos_rosto_data = processa_poses_video(video_data)

    video_data.classificador_blur = analisa_video_rodenas(video_data)
    video_data.classificador_fft = classifica_video_fft(video_data)
    video_data.classificador_pos_rosto = classifica_faces(video_data)
    video_data.classificador_juncao = classificador.predict([[video_data.classificador_pos_rosto, video_data.classificador_fft, video_data.classificador_blur]])

    video_data.frames_otimizados = None
    video_data.frames_originais = None
    video_data.pos_rosto_data = None
    return video_data


def get_labels_videos_deepfake_detection_challeng_metadata():
    with open("../dataset/deepfake-detection-challenge/train_sample_videos/metadata.json") as jsonFile:
        files = json.load(jsonFile)

    file_labels = dict()
    for filename, data in files.items():
        file_labels[filename.split(".")[0]] = data["label"]
    return file_labels

def get_real_videos_names_deepfake_detection_challeng_metadata(file_labels:dict):
    names = list()
    for nome, label in file_labels.items():
        if label == "REAL":
            names.append(nome)
    return names

def get_fake_videos_names_deepfake_detection_challeng_metadata(file_labels:dict):
    names = list()
    for nome, label in file_labels.items():
        if label == "FAKE":
            names.append(nome)
    return names

def executa_em_diretorio_inteiro(diretorio:str):
    file_names = glob.glob(os.path.join(diretorio, '*.mp4'))
    file_paths = [diretorio + x + ".mp4" for x in file_names]
    for i, caminho in enumerate(file_paths):
        try:
            aux = roda_video(caminho, "-")
        except Exception as error:
            print(traceback.format_exc())
            continue


if __name__ == "__main__":
    roda_video("../dataset/deepfake-detection-challenge/train_sample_videos/dlpoieqvfb.mp4", "")
