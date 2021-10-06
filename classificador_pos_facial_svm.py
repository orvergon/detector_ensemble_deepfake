from video_data import VideoData
from posicao_facial import CalculadorPose
import numpy
import pickle
import sklearn.svm._classes

nariz = list(range(28, 37))
sobrancelha_esquerda = list(range(18, 23))
sobrancelha_direita = list(range(23, 28))
cantos_boca = [49, 55]
centro_rosto = nariz + sobrancelha_esquerda + sobrancelha_direita + cantos_boca
face_inteira = list(range(1, 18)) + centro_rosto

def classifica_faces(video_data: VideoData, caminho_classificador="../dataset/deepfake-detection-challenge/train_sample_videos/modelo_SVM.pkl"):
    pose_estimator = CalculadorPose((video_data.largura, video_data.altura))

    with open(caminho_classificador, 'rb') as f:
        model = pickle.load(f, fix_imports=True)
    classifier = model[0]
    scaler = model[1]
    probabilidades = list()
    for face in video_data.pontos_faciais:
        for frame in face:
            probabilidade = examine_a_face(frame, classifier, scaler, pose_estimator)
            probabilidades.append(probabilidade)
    return sum(probabilidades)/len(probabilidades)

def examine_a_face(landmarks, classifier, scaler, pose_estimator: CalculadorPose):
    # extract head pose
    R_c, t_c = None, None
    R_a, t_a = None, None
    R_c_matrix, R_a_matrix = None, None

    R_c, t_c = pose_estimator.solve_single_pose(landmarks, centro_rosto)
    R_a, t_a = pose_estimator.solve_single_pose(landmarks, face_inteira)
    R_c_matrix = pose_estimator.Rodrigues_convert(R_c)
    R_a_matrix = pose_estimator.Rodrigues_convert(R_a)

    rotation_matrix_feature = (R_c_matrix - R_a_matrix).flatten()
    translation_vector_feature = (t_c - t_a)[:, -1]
    feature = numpy.concatenate([rotation_matrix_feature, translation_vector_feature]).reshape(1, -1)
    scaled_feature = scaler.transform(feature)
    score = classifier.predict_proba(scaled_feature)

    return score[0][-1]
