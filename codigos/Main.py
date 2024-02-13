from util.EdgeDetection import EdgeDetector
from util.OtherOperations import Operations
from util.IdentifyAndCountObjects import ObjectDetector

def edge_detection_canny(file_name_or_image):
    if isinstance(file_name_or_image, str):
        canny = EdgeDetector(file_name_or_image)
    else:
        canny = EdgeDetector(None)
    return canny.canny_detector(img=file_name_or_image)

def edge_detection_sobel(file_name_or_image):
    if isinstance(file_name_or_image, str):
        sobel = EdgeDetector(file_name_or_image)
    else:
        sobel = EdgeDetector(None)
    return sobel.sobel_detector(img=file_name_or_image)

def edge_detection_laplacian(file_name_or_image):
    if isinstance(file_name_or_image, str):
        laplacian = EdgeDetector(file_name_or_image)
    else:
        laplacian = EdgeDetector(None)
    return laplacian.filter_laplacian(img=file_name_or_image)


# Função principal
def main():

    file_name = '../images/othergroup.jpg'

    detector = ObjectDetector()

    # Detectar e classificar objetos na imagem
    result, image_with_classification = detector.detect_objects(file_name)

    Operations.display(image_with_classification, "Resultado")


if __name__ == "__main__":
    main()