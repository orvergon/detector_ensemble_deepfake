import cv2

def display_image(image):
    cv2.imshow('image', image)
    while (k := cv2.waitKey(5000)) != 27: #esc
        continue
    cv2.destroyAllWindows()
