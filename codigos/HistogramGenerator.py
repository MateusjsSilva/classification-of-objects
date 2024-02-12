from matplotlib import pyplot as plt
import numpy as np
import cv2

class HistogramAnalyzer:

    def __init__(self, file_name):
        self.file_name = file_name


    def analyze_and_show_gray_histogram(self):
        img = cv2.imread(self.file_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convertendo para P&B
        cv2.imshow("Imagem em Tons de Cinza", img_gray)

        # Função calcHist para calcular o histograma da imagem P&B
        h_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        plt.figure()
        plt.title("Histograma Tons de Cinza")
        plt.xlabel("Intensidade")
        plt.ylabel("Número de Pixels")
        plt.plot(h_gray)
        plt.xlim([0, 256])
        plt.show()


    def analyze_and_show_color_histogram(self):
        img = cv2.imread(self.file_name)

        # Separa os canais de cor
        canais = cv2.split(img)
        cores = ("b", "g", "r")
        plt.figure()
        plt.title("Histograma Colorido")
        plt.xlabel("Intensidade")
        plt.ylabel("Número de Pixels")
        for (canal, cor) in zip(canais, cores):
            # Este loop executa 3 vezes, uma para cada canal
            hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
            plt.plot(hist, color=cor)
        plt.xlim([0, 256])
        plt.show()