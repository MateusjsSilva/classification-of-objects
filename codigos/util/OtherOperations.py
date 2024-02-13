import os
import cv2

class Operations:

    # Função para exibir imagem
    @staticmethod
    def display(image, image_name, max_width=1200, max_height=800):

        # Redimensiona a imagem empilhada para controlar o tamanho máximo da janela
        if  image.shape[1] > max_width or image.shape[0] > max_height:
            scale_factor = min(max_width / image.shape[1], max_height / image.shape[0])
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

        cv2.imshow(image_name, image)
        cv2.waitKey(0)

    @staticmethod
    def save_image(image, save_path):
        
        # Extrair o diretório pai do caminho de salvamento
        directory = os.path.dirname(save_path)
        
        # Verificar se o diretório pai existe e criar, se necessário
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Salvar a imagem
        cv2.imwrite(save_path, image)