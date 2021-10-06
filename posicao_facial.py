import numpy
import cv2
from video_data import VideoData

nariz = list(range(28, 37))
sobrancelha_esquerda = list(range(18, 23))
sobrancelha_direita = list(range(23, 28))
cantos_boca = [49, 55]
centro_rosto = nariz + sobrancelha_esquerda + sobrancelha_direita + cantos_boca
face_inteira = list(range(1, 18)) + centro_rosto


class CalculadorPose():
    altura:int
    largura:int

    def __init__(self, tamanho_img:tuple):
        self.size = tamanho_img
        self.model_points_68 = self._get_full_model_points()
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = numpy.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        self.dist_coeffs = numpy.zeros((4, 1))

    def solve_single_pose(self, marks2D, marks_id):
        marks3D = self.get_pose_marks(self.model_points_68, marks_id)
        marks2D = self.get_pose_marks(marks2D, marks_id)

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            marks3D,
            marks2D,
            self.camera_matrix, self.dist_coeffs)

        return (rotation_vector, translation_vector)

    def get_pose_marks(self, marks, markID):
        chosen_marks = []
        for ID in markID:
            chosen_marks.append(marks[ID - 1])
        return numpy.array(chosen_marks, dtype=numpy.float32)

    def _get_full_model_points(self):
        raw_value = []
        with open("../lib/headpose_forensic/models/model_landmark.txt") as file:
            for line in file:
                raw_value.append(line)
        model_points = numpy.array(raw_value, dtype=numpy.float32)
        model_points = numpy.reshape(model_points, (3, -1)).T
        model_points[:, -1] *= -1
        return numpy.array(model_points, dtype=numpy.float32)

    def Rodrigues_convert(self, R):
        R = cv2.Rodrigues(R)[0]
        return R

def processa_poses_video(video_data:VideoData):
    video_dimentions = (video_data.largura,
                        video_data.altura)
    encontra_pose = CalculadorPose(video_dimentions)
    R_c_list, R_a_list, t_c_list, t_a_list = [], [], [], []
    R_c_matrix_list, R_a_matrix_list = [], []
    aux_dict = dict()
    for face in video_data.pontos_faciais:
        for frame in face:
            R_c, t_c = encontra_pose.solve_single_pose(frame, centro_rosto)
            R_a, t_a = encontra_pose.solve_single_pose(frame, face_inteira)

            R_c_matrix = encontra_pose.Rodrigues_convert(R_c)
            R_a_matrix = encontra_pose.Rodrigues_convert(R_a)

            R_c_list.append(R_c)
            R_a_list.append(R_a)

            t_c_list.append(t_c)
            t_a_list.append(t_a)

            R_c_matrix_list.append(R_c_matrix)
            R_a_matrix_list.append(R_a_matrix)

    aux_dict['R_c_vec'] = R_c_list
    aux_dict['R_c_mat'] = R_c_matrix_list
    aux_dict['t_c'] = t_c_list

    aux_dict['R_a_vec'] = R_a_list
    aux_dict['R_a_mat'] = R_a_matrix_list
    aux_dict['t_a'] = t_a_list

    return aux_dict

def classifica_frames(video_data):
    pass