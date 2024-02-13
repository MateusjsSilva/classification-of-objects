
import numpy as np
import cv2

def remove_shadow(imagem):
    _, mascara = cv2.threshold(imagem, 224, 255, cv2.THRESH_BINARY_INV)
    elemEst = np.ones((5, 5), np.uint8)
    mascara_sombra = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemEst)
    img = cv2.addWeighted(imagem, 1, mascara_sombra, 0.2, 0)
    return img


def processamento(imagem):
    imgPre = cv2.GaussianBlur(imagem, (5, 5), 1.5)
    elemEst = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_OPEN, elemEst)
    imgPre =  cv2.Canny(imgPre, 0, 255) #edge_detection_sobel(imagem)
    contornos, _ = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 0, 0), 2) 
    return imagem


def limites(imagem_original, imagem_borda):
    contador = [0, 0, 0]
    imagem_borda = cv2.cvtColor(imagem_borda, cv2.COLOR_BGR2GRAY)
    contornos, _ = cv2.findContours(imagem_borda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        cv2.putText(imagem_original, texto, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
