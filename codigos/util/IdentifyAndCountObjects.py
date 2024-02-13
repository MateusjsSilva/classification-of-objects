import cv2
import numpy as np

class ObjectDetector:

    def detect_objects(self, image_or_path):

        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        else:
            image = image_or_path
        
        # Segmentar objetos por cor
        color_ranges = {
            "vermelho": ([0, 100, 100], [10, 255, 255]),  # Definir intervalo de cor para vermelho
            "verde": ([36, 25, 25], [70, 255, 255]),      # Definir intervalo de cor para verde
            "azul": ([100, 100, 100], [140, 255, 255])   # Definir intervalo de cor para azul
        }
        segmented_objects = self.segment_by_color(image, color_ranges)

        # Segmentar objetos por forma
        shape_segmentation = self.segment_by_shape(image)

        # Detectar objetos na imagem segmentada por forma
        objects_detected, image_with_classification = self.detect_objects_from_segmented_image(shape_segmentation)

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

        return segmented_objects

    def segment_by_shape(self, image):
        # Converter a imagem para tons de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicar operações morfológicas para suavizar as bordas
        kernel = np.ones((5,5), np.uint8)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Detecção de bordas
        edges = cv2.Canny(closing, 100, 200)

        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Desenhar contornos na imagem original
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result

    def detect_objects_from_segmented_image(self, segmented_image):
        objects_detected = {}

        # Converter a imagem segmentada para tons de cinza
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

        # Detecção de bordas
        edges = cv2.Canny(gray, 100, 200)

        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Desenhar contornos na imagem segmentada
        image_with_classification = segmented_image.copy()
        for i, contour in enumerate(contours, start=1):
            # Calcular o centro do contorno
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Desenhar o texto com a classificação
                cv2.putText(image_with_classification, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        # Contar objetos
        objects_detected = len(contours)

        return objects_detected, image_with_classification