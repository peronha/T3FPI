import numpy as np
import cv2

def calculateBlur(x):
    blur = x * 3;

    if blur % 2 == 0:
        blur -= 1

    if blur <= 0:
        blur = 1

    return blur

def nothing(x):

    pass


def pollChanges(frame):

    blur_value = cv2.getTrackbarPos('Blur', 'Efeitos')
    blur_kernel = calculateBlur(blur_value)

    is_canny = cv2.getTrackbarPos('Canny', 'Efeitos')
    is_grayscale = cv2.getTrackbarPos('Cinza', 'Efeitos')
    is_resize = cv2.getTrackbarPos('Reduzir', 'Efeitos')
    is_negative = cv2.getTrackbarPos('Negativo', 'Efeitos')
    contrast_value = cv2.getTrackbarPos('Contraste', 'Efeitos')
    brightness_value = cv2.getTrackbarPos('Brilho', 'Efeitos')

    frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0);

    if is_canny == 1:
        frame = cv2.Canny(frame, 100, 100)

    if is_grayscale == 1 and is_canny != 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if is_negative == 1:
        frame = cv2.bitwise_not(frame)

    if is_resize == 1:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    if contrast_value > 0:
        frame = cv2.addWeighted(frame, contrast_value + 1, np.zeros(frame.shape, frame.dtype), 0, brightness_value)

    if brightness_value > 0:
        frame = cv2.addWeighted(frame, 1, np.zeros(frame.shape, frame.dtype), 0, brightness_value)

    return frame

def getFrame(frame):


    return pollChanges(frame)


def criaJanelaEfeitos():
    cv2.namedWindow('Efeitos')

    cv2.createTrackbar('Blur', 'Efeitos', 0, 10, nothing)
    cv2.createTrackbar('Canny', 'Efeitos', 0, 1, nothing)
    cv2.createTrackbar('Cinza', 'Efeitos', 0, 1, nothing)
    cv2.createTrackbar('Reduzir', 'Efeitos', 0, 1, nothing)
    cv2.createTrackbar('Negativo', 'Efeitos', 0, 1, nothing)
    cv2.createTrackbar('Contraste', 'Efeitos', 0, 5, nothing)
    cv2.createTrackbar('Brilho', 'Efeitos', 0, 100, nothing)

def __main__():

    cap = cv2.VideoCapture(0)

    criaJanelaEfeitos()

    out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        new_frame = getFrame(frame)

        cv2.imshow('Captura de Video',new_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

__main__()