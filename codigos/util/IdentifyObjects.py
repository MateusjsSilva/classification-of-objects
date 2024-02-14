import cv2
import numpy as np
from sklearn.cluster import KMeans

from util.OtherOperations import Operations

class ObjectDetector:

    def process_image(self, image_or_path):
        """
        Processa a imagem e retorna objetos classificados.
        """
        # Carrega a imagem
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        else:
            image = image_or_path

        # Remove sombra
        image_not_shadow = self.remove_shadow(image)
        
        # Segmenta por cor
        color_ranges = {
            "vermelho": ([0, 100, 100], [10, 255, 255]),
            "verde": ([36, 25, 25], [70, 255, 255]),
            "azul": ([100, 100, 100], [140, 255, 255])
        }
        segmented_objects = self.segment_by_color(image_not_shadow, color_ranges)
        
        # Identificar cores dominantes na imagem
        centroids = self.find_color_clusters(image_not_shadow)  # Ajuste para usar a imagem correta

        # Segmenta por forma
        shape_segmentation = self.segment_by_shape(image_not_shadow)
        
        # Agora 'objects_detected' deve ser a lista de contornos.
        objects_detected, image_with_classification, count_objects = self.detect_objects_from_segmented_image(shape_segmentation, image)

        # E então passamos essa lista para 'classify_objects'
        classified_objects, image_classified = self.classify_objects(objects_detected, image, centroids)

        return image_classified, classified_objects, image_with_classification, count_objects

    #
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
    
    #
    def segment_by_shape(self, image):

        # Converter a imagem para tons de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Suavização usando filtro Gaussiano
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        #Operations.display(gray, "Suavizacao com Blur")
        #Operations.save_image(gray, 'cache/image_gray_soft_blur.jpg')
        
        # Aplicar operações morfológicas para suavizar as bordas e corrigir objetos justapostos
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Aplicar uma erosão mais forte nos objetos conectados
        # Você pode precisar ajustar o tamanho do kernel e o número de iterações
        kernel = np.ones((5, 5), np.uint8)  # Tamanho do kernel aumentado para erosão mais forte
        eroded = cv2.erode(opening, kernel, iterations=2)
        
        # Após a erosão, dilatar com um kernel menor
        kernel = np.ones((3, 3), np.uint8)  # Kernel menor para dilatação
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        # Detecção de bordas
        edges = cv2.Canny(dilated, 5, 130)

        #Operations.display(edges, "Suavizacao com Blur")

        # Aplicar dilatação para unir áreas próximas e corrigir bordas mal definidas
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Encontrar contornos após a dilatação
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Criar uma imagem em branco com as mesmas dimensões que a imagem de entrada
        result = np.zeros_like(image)

        # Desenhar contornos na imagem em branco
        cv2.drawContours(result, contours, -1, (255, 255, 255), 2)

        return result

    #
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
        count_objects_detected = len(contours)

        return contours, image_with_classification, count_objects_detected
    
    #
    def find_color_clusters(self, image, n_clusters=3):
        """
        Identifica os clusters de cor na imagem usando o algoritmo K-Means.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_values = hsv_image[:, :, 0].reshape(-1, 1)

        # Aplica o K-Means ao canal Hue
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(hue_values)
        centroids = kmeans.cluster_centers_

        return centroids

    #
    def classify_objects(self, contours, image, centroids):
        """
        Classifica objetos com base em tamanho, forma e cor, usando os centroids dos clusters de cor.
        """
        classified_objects = []

        for contour in contours:
            properties = {}
            
            # Calcula a área do contorno para classificação de tamanho
            area = cv2.contourArea(contour)
            properties['size'] = 'Small' if area < 1000 else 'Large'  # Limiar arbitrário
            
            # Calcula o retângulo delimitador e o círculo mínimo envolvente para classificação de forma
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            properties['shape'] = 'Circle' if 0.9 < aspect_ratio < 1.1 else 'Rectangle'
            
            # Calcula o valor médio de Hue dentro do contorno
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            object_hue_mean = self.get_object_hue_mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), mask)

            # Classifica a cor com base nos centroids de cor
            color_classification = self.classify_colors(object_hue_mean, centroids)
            properties['color'] = f"{round(color_classification, 1)}"
            
            # Adiciona propriedades classificadas à lista
            classified_objects.append(properties)
            
            # Opcional: Desenha a classificação na imagem
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                label = f"{properties['color']}, {properties['shape']}, {properties['size']}"
                cv2.putText(image, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        return classified_objects, image
    
    #
    def get_object_hue_mean(self, hsv_image, mask):
        """
        Calcula o valor médio de Hue para o objeto usando a máscara do contorno.
        """
        hue_values = hsv_image[:,:,0][mask == 255]
        return np.mean(hue_values)  
    
    #
    def classify_colors(self, hue_value, centroids):
        """
        Identifica a cor mais próxima do valor médio de Hue do objeto com base nos centroids.
        """
        closest_centroid = min(centroids, key=lambda x: abs(x[0] - hue_value))
        return closest_centroid[0]

    #
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