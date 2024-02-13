import cv2
import numpy as np

from util.OtherOperations import Operations

class ObjectDetector:

    def detect_objects(self, image_or_path):

        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        else:
            image = image_or_path

        image_not_shadow = ObjectDetector.remove_shadow(image)

        #Operations.display(image_not_shadow, "Imagem sem sombra.")
        #Operations.save_image(image_not_shadow, 'cache/image_not_shadow.jpg')
        
        # Segmentar objetos por cor
        color_ranges = {
            "marrom": ([26, 16, 175], [46, 36, 195]),   # Definir intervalo de cor para marrom
            "vermelho": ([0, 100, 100], [10, 255, 255]),  # Definir intervalo de cor para vermelho
            "verde": ([36, 25, 25], [70, 255, 255]),      # Definir intervalo de cor para verde
            "azul": ([100, 100, 100], [140, 255, 255])   # Definir intervalo de cor para azul
        }
        segmented_objects = self.segment_by_color(image_not_shadow, color_ranges)

        # Segmentar objetos por forma
        shape_segmentation = self.segment_by_shape(image_not_shadow)

        #Operations.display(shape_segmentation, "Segamentação por forma")

        # Detectar objetos na imagem segmentada por forma
        objects_detected, image_with_classification = self.detect_objects_from_segmented_image(shape_segmentation, image)

        return objects_detected, image_with_classification

    def segment_by_color(self, image, color_ranges):
        # Converter a imagem para o espaço de cores HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        segmented_objects = {}

        # Segmentar objetos para cada intervalo de cor
        for color_name, (lower_color, upper_color) in color_ranges.items():
            # Converter os valores para numpy array
            lower_color = np.array(lower_color)
            upper_color = np.array(upper_color)
            mask = cv2.inRange(hsv_image, lower_color, upper_color)
            segmented_objects[color_name] = cv2.bitwise_and(image, image, mask=mask)

        # Suponhamos que os objetos marrons estão no intervalo 'vermelho' por simplicidade
        brown_objects_mask = segmented_objects["marrom"]

        # Aplicar Watershed para separar objetos justapostos
        # Primeiro, converta a máscara para cinza e binarize-a
        gray = cv2.cvtColor(brown_objects_mask, cv2.COLOR_BGR2GRAY)
        ret, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Removendo ruído com abertura
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Encontrar área de fundo com dilatação
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Encontrar área de primeiro plano seguro
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Encontrar área desconhecida (bordas entre objetos)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Rotular os componentes conectados
        ret, markers = cv2.connectedComponents(sure_fg)

        # Incrementar todos os marcadores em 1 e marcar área desconhecida com 0
        markers = markers + 1
        markers[unknown == 255] = 0

        # Aplicar Watershed
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [0, 255, 0]  # As bordas dos objetos ficarão verdes

        segmented_objects['watershed'] = image

        return segmented_objects
    
    def segment_by_shape(self, image):
        # Converter a imagem para tons de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Suavização usando filtro Gaussiano
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        #Operations.display(gray, "Suavização com Blur")
        #Operations.save_image(gray, 'cache/image_gray_soft.jpg')
        
        # Aplicar operações morfológicas para suavizar as bordas e corrigir objetos justapostos
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Detecção de bordas
        edges = cv2.Canny(opening, 20, 180)

        # Aplicar dilatação para unir áreas próximas e corrigir bordas mal definidas
        dilation = cv2.dilate(edges, kernel, iterations=1)

        # Encontrar contornos após a dilatação
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Criar uma imagem em branco com as mesmas dimensões que a imagem de entrada
        result = np.zeros_like(image)

        # Desenhar contornos na imagem em branco
        cv2.drawContours(result, contours, -1, (255, 255, 255), 2)

        return result

    def detect_objects_from_segmented_image(self, segmented_image, original_image):
        # Converter a imagem segmentada para tons de cinza
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

        # Detectar contornos
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Desenhar contornos na imagem original
        image_with_classification = original_image.copy()
        for i, contour in enumerate(contours, start=1):
            # Calcular a área do contorno
            area = cv2.contourArea(contour)
            if area > 100:  # Apenas considerar contornos com uma área mínima
                # Calcular o centro do contorno
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Desenhar o texto com a classificação
                    cv2.putText(image_with_classification, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)
                    
                    # Desenhar o contorno do objeto
                    cv2.drawContours(image_with_classification, [contour], -1, (0, 255, 0), 2)

        # Contar objetos
        objects_detected = len(contours)

        return objects_detected, image_with_classification
    
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