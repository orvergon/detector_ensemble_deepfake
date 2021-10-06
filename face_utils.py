import numpy

def obtem_todos_pontos_modelo3D(self):
    raw_value = []
    with open("../lib/headpose_forensic/models/model_landmark.txt") as file:
        for line in file:
            raw_value.append(line)
    model_points = numpy.array(raw_value, dtype=numpy.float32)
    model_points = numpy.reshape(model_points, (3, -1)).T
    model_points[:, -1] *= -1
    return numpy.array(model_points, dtype=numpy.float32)


def obtem_subset_pontos(self, pontos, ids_pontos):
    chosen_marks = []
    for id in ids_pontos:
        chosen_marks.append(pontos[id - 1])
    return numpy.array(chosen_marks, dtype=numpy.float32)
