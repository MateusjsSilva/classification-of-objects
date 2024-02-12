import numpy as np
import cv2
from skimage.filters import threshold_local

class ImageProcessor:

    def __init__(self, file_name):
        self.file_name = file_name

    # Aplicar operações de pré-processamento, como suavização, binarização, etc.
    def preprocess_image(self):
        imgColorida = cv2.imread(self.file_name)
        img = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Imagem Original", self.remove_shadow(img))
        cv2.waitKey(0)
        return 0

    def remove_shadow(self, imagem):
        _, máscara = cv2.threshold(imagem, 220, 255, cv2.THRESH_BINARY_INV)
        elemEst = np.ones((5, 5), np.uint8)
        máscara_sombra = cv2.morphologyEx(máscara, cv2.MORPH_OPEN, elemEst)
        img = cv2.addWeighted(imagem, 1, máscara_sombra, 0.3, 0)
        return img

    def process_and_show_results(self):

        # Função para facilitar a escrita nas imagem
        def escreve(img, texto, cor=(255,0,0)):
            fonte = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, texto, (10,20), fonte, 0.5, cor, 0, cv2.LINE_AA)

        imgColorida = cv2.imread(self.file_name)

         # Passo 1: Remover sombra
        imgSemSombra = self.remove_shadow(imgColorida)

        # Passo 1: Conversão para tons de cinza
        img = cv2.cvtColor(imgSemSombra, cv2.COLOR_BGR2GRAY)

        # Passo 2: Blur/Suavização da imagem
        suave = cv2.blur(img, (7, 7))

        # Passo 3: Binarização resultando em pixels brancos e pretos
        T = threshold_local(suave, block_size=11, offset=10)
        bin = (suave > T).astype("uint8") * 255

        # Passo 4: Detecção de bordas com Canny
        bordas = cv2.Canny(bin, 70, 150)
        
        # Passo 5: Identificação e contagem dos contornos da imagem
        # cv2.RETR_EXTERNAL = conta apenas os contornos externos
        objetos, _ = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        escreve(img, "Imagem em tons de cinza", 0)
        escreve(suave, "Suavizacao com Blur", 0)
        escreve(bin, "Binarizacao com Threshold Local Adaptativo", 255)
        escreve(bordas, "Detector de bordas Canny", 255)
        temp = np.vstack([
            np.hstack([img, suave]),
            np.hstack([bin, bordas])
        ])
        cv2.imshow("Quantidade de objetos: "+str(len(objetos)), temp)
        cv2.waitKey(0)
        imgC2 = imgColorida.copy()
        cv2.imshow("Imagem Original", imgColorida)
        cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2)
        escreve(imgC2, str(len(objetos))+" objetos encontrados!")
        cv2.imshow("Resultado", imgC2)
        cv2.waitKey(0)
