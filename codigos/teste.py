import cv2
import numpy as np
from matplotlib import pyplot as plt

contador = [0, 0, 0]

def limites(imagem_original=cv2.Mat, imagem_borda=cv2.Mat):
    contornos, _ = cv2.findContours(
        imagem_borda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)

        sub = imagem_original[y:y+h, x:x+w]
        color = ('b', 'g', 'r')
        médias = []
        variânças = []
        for i, col in enumerate(color):
            histograma = cv2.calcHist([sub], [i], None, [256], [0, 256])
            k = np.mean(sub[:, :, i])
            v = np.var(sub[:, :, i])
            médias.append(k)
            variânças.append(v)
            plt.plot(histograma, color=col)
            plt.xlim([0, 256])
        m = max(médias)
        v = médias[2] - médias[1]
        texto = "0"
        if m < 100:
            texto = "pimenta "
            contador[0] += 1
        elif v < 26:
            texto = "ervilha "
            contador[1] += 1
        else:
            texto = "feijao "
            contador[2] += 1    
        plt.savefig(texto + ".png")
        plt.clf()
        cv2.putText(imagem_original, texto, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    exibir(imagem_original)


def sombra(imagem=cv2.Mat) -> cv2.Mat:
    _, máscara = cv2.threshold(imagem, 224, 255, cv2.THRESH_BINARY_INV)
    elemEst = np.ones((7, 7), np.uint8)
    máscara_sombra = cv2.morphologyEx(máscara, cv2.MORPH_OPEN, elemEst)
    img = cv2.addWeighted(imagem, 1.1, máscara_sombra, 0.1, 0)
    return img


def processamento(imagem=cv2.Mat) -> cv2.Mat:
    imgPre = cv2.GaussianBlur(imagem, (7, 7), 0)

    elemEst = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_OPEN, elemEst)

    imgPre = cv2.Canny(imgPre, 20, 180)
    contornos, _ = cv2.findContours(
        imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 0, 0), 2)
    return imgPre


def exibir(imagem=cv2.Mat) -> None:
    cv2.namedWindow("My Image", cv2.WINDOW_NORMAL)

    altura = 800
    lagura = 600
    cv2.resizeWindow("My Image", lagura, altura)

    cv2.imshow("My Image", imagem)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img_cor = cv2.imread("../images/img3.jpg")
    img = cv2.imread('../images/img3.jpg', cv2.IMREAD_GRAYSCALE)
    img = sombra(img)
    exibir(img)
    img = processamento(img)
    exibir(img)
    limites(img_cor, img)


main()
