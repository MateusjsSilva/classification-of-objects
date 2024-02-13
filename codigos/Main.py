from util.OtherOperations import Operations
from util.IdentifyAndCountObjects import ObjectDetector

# Função principal
def main():

    file_name = '../images/img3.jpg'

    detector = ObjectDetector()

    # Detectar e classificar objetos na imagem
    result, image_with_classification = detector.detect_objects(file_name)

    Operations.display(image_with_classification, "Resultado")


if __name__ == "__main__":
    main()