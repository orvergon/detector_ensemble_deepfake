import cv2


def otimiza_frames(frame_list: list):
    optimized_frames = list()
    for index, frame in enumerate(frame_list):
        new_frame = otimiza_frame(frame)
        optimized_frames.append(new_frame)
    return optimized_frames


def otimiza_frame(frame):
    new_frame = downsize_frame(frame, 0.7)
    new_frame = grayscale(new_frame)
    return new_frame


def grayscale(frame):
    frame = frame[:, :, ::-1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def downsize_frame(frame, downsize_factor: float):
    new_shape = (int(frame.shape[1] * downsize_factor), int(frame.shape[0] * downsize_factor))
    frame = cv2.resize(frame, new_shape)
    return frame
