import cv2
import numpy as np
from scipy import stats

from util.OtherOperations import Operations

class ObjectDetector:

    # Processa a imagem e retorna objetos classificados.
    def process_image(self, image_or_path):
   
        # Carrega a imagem
        image = cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path

        # Remove sombra
        image_not_shadow = self.remove_shadow(image)
        
        # Segmenta por forma
        shape_segmentation = self.segment_by_shape(image_not_shadow)
        
        # Agora 'objects_detected' deve ser a lista de contornos.
        objects_detected, image_with_classification, count_objects = self.detect_objects_from_segmented_image(shape_segmentation, image)

        # E então passamos essa lista para 'classify_objects'
        classified_objects, image_classified = self.classify_objects(objects_detected, image_with_classification)

        return image_classified, classified_objects, image_with_classification, count_objects

    def segment_by_shape(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Aplicar operações morfológicas para suavizar as bordas e corrigir objetos justapostos
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        
        # Detecção de bordas
        edges = cv2.Canny(opening, 5, 130)

        #Operations.display(edges, '')
        Operations.save_image(edges, '../resources/results/edges.jpg')

        # Aplicar dilatação para unir áreas próximas e corrigir bordas mal definidas
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Encontrar contornos após a dilatação
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Criar uma imagem em branco com as mesmas dimensões que a imagem de entrada
        result = np.zeros_like(image)

        # Desenhar contornos na imagem em branco
        cv2.drawContours(result, contours, -1, (255, 255, 255), 2)

        return result

    def detect_objects_from_segmented_image(self, segmented_image, original_image):

        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

        # Detectar contornos
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Desenhar contornos na imagem original
        image_with_classification = original_image.copy()
        for i, contour in enumerate(contours, start=1):
            # Calcular a área do contorno
            area = cv2.contourArea(contour)
            if area > 100:  # Apenas considerar contornos com uma área mínima
                 # Desenhar o contorno do objeto
                cv2.drawContours(image_with_classification, [contour], -1, (0, 255, 0), 2)

        return contours, image_with_classification, len(contours)
    

    def classify_objects(self, contours, image):
        """
        Classifica objetos com base em tamanho, forma e cor, usando os centroids dos clusters de cor.
        """
        classified_objects = []

        for contour in contours:
            properties = {}

            # Aproxima o contorno com precisão
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # Classifica a forma com base no número de vértices
            vertices = len(approx)
            shape = "unidentified"
            if vertices == 3:
                shape = "Triangle"
            elif vertices == 4:
                # Para diferenciar retângulo de quadrado
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                shape = "Square" if 0.9 < aspect_ratio < 1.1 else "Rectangle"
            elif vertices > 4:
                shape = "Circle"

            properties['shape'] = shape
            
            # Calcula a cor predominante
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            object_hsv_dominant = self.get_object_hsv_dominant(hsv_image, mask)
            properties['color'] = self.classify_colors(object_hsv_dominant)

            classified_objects.append(properties)
            
            # Opcional: Desenha a classificação na imagem
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                label = f"{properties['color']}, {properties['shape']}"
                cv2.putText(image, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (247, 137, 47), 2)
                
        return classified_objects, image

    def identify_shape(self, vertices, approx):
        if vertices == 3:
            return "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            return "Square" if 0.9 < aspect_ratio < 1.1 else "Rectangle"
        elif vertices > 4:
            return "Circle"
        return "unidentified"

    def get_object_hsv_dominant(self, hsv_image, mask):
        """
        Calcula a cor HSV predominante para o objeto usando a máscara do contorno
        baseando-se no histograma do canal Hue.
        """
        masked_hsv = hsv_image[mask == 255]
        if masked_hsv.size == 0:
            return (0, 0, 0)  # Retorna uma cor padrão se não houver pixels dentro do contorno

        # Calcula o histograma do canal Hue
        hue_histogram = cv2.calcHist([masked_hsv], [0], None, [180], [0, 180])
        hue_histogram = hue_histogram.ravel() / hue_histogram.sum()
        hue_max = np.argmax(hue_histogram)

        # Calcula a média dos valores de Saturação e Brilho (Value) para o intervalo de Hue mais comum
        relevant_pixels = masked_hsv[(masked_hsv[:, 0] >= hue_max - 2) & (masked_hsv[:, 0] <= hue_max + 2)]
        if relevant_pixels.size == 0:
            return (hue_max, 0, 0)  # Retorna o Hue predominante com Saturação e Brilho mínimos

        saturation_mean = np.mean(relevant_pixels[:, 1])
        value_mean = np.mean(relevant_pixels[:, 2])

        return (hue_max, saturation_mean, value_mean)

    def classify_colors(self, hsv_value):
        hue, saturation, value = hsv_value

        if value < 50 or saturation < 50:
            return 'Black'
        elif 5 <= hue <= 18:  # Intervalo para Laranja
            return 'Orange'
        elif 22 <= hue <= 38 and saturation > 50 and value > 50:  # Intervalo expandido para Amarelo
            return 'Yellow'
        elif 39 <= hue <= 80:  # Intervalo ajustado para Verde, começando após o Amarelo
            return 'Green'
        elif (0 <= hue <= 10 or 170 <= hue <= 180) and saturation > 50 and value > 50:
            return 'Red'
        elif 105 <= hue <= 135:  # Intervalo para Azul
            return 'Blue'
        elif 140 <= hue <= 160:  # Intervalo para Roxo
            return 'Purple'
        else:
            return "Undefined"
    
    @staticmethod
    def remove_shadow(imagem):

        # Aplicar limiarização para segmentar a sombra da imagem
        _, mascara = cv2.threshold(imagem, 224, 255, cv2.THRESH_BINARY_INV)
        elemEst = np.ones((7, 7), np.uint8)
        
        # Aplicar abertura morfológica para remover pequenos ruídos e suavizar a sombra
        mascara_sombra = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemEst)
        
        # Adicionar a sombra removida à imagem original com uma leve transparência
        img = cv2.addWeighted(imagem, 1.1, mascara_sombra, 0.1, 0)
        return img