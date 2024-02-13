import numpy as np
import cv2

class EdgeDetector:

    def __init__(self, file_name):
        self.file_name = file_name


    def filter_laplacian(self, img=None, show_results=False, write_text=False):
        """
        Detecta bordas usando o filtro Laplaciano.

        :param img: Imagem de entrada em tons de cinza (default: None)
        :param show_results: Se True, os resultados serão exibidos (default: True)
        :param write_text: Se True, escreve na imagem (default: False)
        :return: Imagem com as bordas detectadas
        """
        if img is None:
            img = cv2.imread(self.file_name, cv2.IMREAD_GRAYSCALE)

        lap = cv2.Laplacian(img, cv2.CV_64F)
        lap = np.uint8(np.absolute(lap))

        resultado = np.vstack([img, lap])

        if write_text:
            escreve(resultado, "Filtro Laplaciano")

        if show_results:
            max_width = 1200
            max_height = 800
            if resultado.shape[1] > max_width or resultado.shape[0] > max_height:
                scale_factor = min(max_width / resultado.shape[1], max_height / resultado.shape[0])
                resultado = cv2.resize(resultado, None, fx=scale_factor, fy=scale_factor)

            cv2.imshow("Filtro Laplaciano", resultado)
            cv2.waitKey(0)

        return resultado


    def sobel_detector(self, img=None, show_results=False, write_text=False):
        """
        Detecta bordas usando o operador Sobel.

        :param img: Imagem de entrada em tons de cinza (default: None)
        :param show_results: Se True, os resultados serão exibidos (default: True)
        :param write_text: Se True, escreve na imagem (default: False)
        :return: Imagem com as bordas detectadas
        """
        if img is None:
            img = cv2.imread(self.file_name, cv2.IMREAD_GRAYSCALE)

        suave = cv2.GaussianBlur(img, (5, 5), 1.5)
        sobelX = cv2.Sobel(suave, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(suave, cv2.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobel = cv2.bitwise_or(sobelX, sobelY)

        if write_text:
            escreve(img, "Imagem em tons de cinza")
            escreve(sobelX, "Sobel X")
            escreve(sobelY, "Sobel Y")
            escreve(sobel, "Sobel")

        resultado = np.vstack([
            np.hstack([img, sobelX]),
            np.hstack([sobelY, sobel])
        ])

        if show_results:
            max_width = 1200
            max_height = 800
            if resultado.shape[1] > max_width or resultado.shape[0] > max_height:
                scale_factor = min(max_width / resultado.shape[1], max_height / resultado.shape[0])
                resultado = cv2.resize(resultado, None, fx=scale_factor, fy=scale_factor)

            cv2.imshow("Sobel", resultado)
            cv2.waitKey(0)

        return sobel


    def canny_detector(self, img=None, show_results=False, write_text=False):
        """
        Detecta bordas usando o detector de Canny.

        :param img: Imagem de entrada em tons de cinza (default: None)
        :param show_results: Se True, os resultados serão exibidos (default: True)
        :param write_text: Se True, escreve na imagem (default: False)
        :return: Imagem com as bordas detectadas
        """
        if img is None:
            img = cv2.imread(self.file_name, cv2.IMREAD_GRAYSCALE)

        suave = cv2.GaussianBlur(img, (5, 5), 1.5)
        canny1 = cv2.Canny(suave, 20, 200)
        canny2 = cv2.Canny(suave, 20, 120)

        if write_text:
            escreve(img, "Imagem em tons de cinza")
            escreve(suave, "Suavizacao com Blur")
            escreve(canny1, "Canny com limiar 1")
            escreve(canny2, "Canny com limiar 2")

        resultado = np.vstack([
            np.hstack([img, suave]),
            np.hstack([canny1, canny2])
        ])

        if show_results:
            max_width = 1200
            max_height = 800
            if resultado.shape[1] > max_width or resultado.shape[0] > max_height:
                scale_factor = min(max_width / resultado.shape[1], max_height / resultado.shape[0])
                resultado = cv2.resize(resultado, None, fx=scale_factor, fy=scale_factor)

            cv2.imshow("Detector de Bordas Canny", resultado)
            cv2.waitKey(0)

        return resultado


# Função para facilitar a escrita nas imagem
def escreve(img, texto, cor=(255, 0, 0), espessura=2):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (40, 60), fonte, 1.5, cor, espessura, cv2.LINE_AA)