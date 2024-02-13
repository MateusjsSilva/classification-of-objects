import cv2
import numpy as np
from matplotlib import pyplot as plt


def remove_shadow(imagem):

    # Aplicar limiarização para segmentar a sombra da imagem
    _, mascara = cv2.threshold(imagem, 220, 255, cv2.THRESH_BINARY_INV)
    elemEst = np.ones((5, 5), np.uint8)
    
    # Aplicar abertura morfológica para remover pequenos ruídos e suavizar a sombra
    mascara_sombra = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemEst)
    
    # Adicionar a sombra removida à imagem original com uma leve transparência
    img = cv2.addWeighted(imagem, 1.1, mascara_sombra, 0.1, 0)
    return img


# Load the image
image_path = '../images/othergroup.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = remove_shadow(gray)

# Apply GaussianBlur, which is useful for removing noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding to get binary image
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours from the binary image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contoured_image = image.copy()
cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)

# Show the original and contoured images
plt.figure(figsize=(10, 10))

plt.imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Contours')

plt.show()