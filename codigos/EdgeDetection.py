import numpy as np
import cv2

class EdgeDetector:

    def __init__(self, file_name):
        self.file_name = file_name

    def filter_laplacian(self):
        img = cv2.imread(self.file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convertendo para P&B

        # 
        lap = cv2.Laplacian(img, cv2.CV_64F)
        lap = np.uint8(np.absolute(lap))
        resultado = np.vstack([img, lap])
        cv2.imshow("Filtro Laplaciano", resultado)
        cv2.waitKey(0)


    def sobel_detector(self):
        img = cv2.imread(self.file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convertendo para P&B

        # 
        sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobel = cv2.bitwise_or(sobelX, sobelY)
        resultado = np.vstack([
            np.hstack([img, sobelX]),
            np.hstack([sobelY, sobel])
            ])
        cv2.imshow("Sobel", resultado)
        cv2.waitKey(0)

    def canny_detector(self):
        img = cv2.imread(self.file_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convertendo para P&B

        # 
        suave = cv2.GaussianBlur(img, (7, 7), 0)
        canny1 = cv2.Canny(suave, 20, 120)
        canny2 = cv2.Canny(suave, 70, 200)
        resultado = np.vstack([
            np.hstack([img, suave ]),
            np.hstack([canny1, canny2])
            ])
        cv2.imshow("Detector de Bordas Canny", resultado)
        cv2.waitKey(0)