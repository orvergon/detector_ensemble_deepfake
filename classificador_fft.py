import numpy
from scipy.interpolate import griddata
from video_data import VideoData
from otimizacao_frame import grayscale
import pickle
import image_utils

epsilon = 1e-8
N = 300

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


def classifica_video_fft(video_data:VideoData):
    flat_landmarks = flat_pontos_faciais(video_data.pontos_faciais)
    dados = numpy.zeros([len(flat_landmarks), N])
    for i, landmarks in enumerate(flat_landmarks):
        bouding_box = get_bouding_box(landmarks)
        face_crop = video_data.frames_originais[i][bouding_box[1][1]:bouding_box[0][1], bouding_box[1][0]:bouding_box[0][0]]
        gray_face = grayscale(face_crop)
        dados[i] = fft_azimuthaly(gray_face)
    with open("trained_svm.pkl", "rb") as svm_pickle:
        svm = pickle.load(svm_pickle)
    dados = svm.predict(dados)
    return sum(dados)/len(dados)

def fft_azimuthaly(frame):
    f = numpy.fft.fft2(frame)
    fshift = numpy.fft.fftshift(f)
    fshift += epsilon

    magnitude_spectrum = 20 * numpy.log(numpy.abs(fshift))

    psd1D = azimuthalAverage(magnitude_spectrum)

    points = numpy.linspace(0, N, num=psd1D.size)
    xi = numpy.linspace(0, N, num=N)

    interpolated = griddata(points, psd1D, xi, method='cubic')
    interpolated /= interpolated[0]
    return interpolated


def azimuthalAverage(image, center=None):
    y, x = numpy.indices(image.shape)

    if not center:
        center = numpy.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = numpy.hypot(x - center[0], y - center[1])

    ind = numpy.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    r_int = r_sorted.astype(int)

    deltar = r_int[1:] - r_int[:-1]
    rind = numpy.where(deltar)[0]
    nr = rind[1:] - rind[:-1]

    csim = numpy.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
