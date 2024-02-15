from util.OtherOperations import Operations
from util.IdentifyObjects import ObjectDetector
from util.HistogramOperations import HistogramProcessor as Hist

# Função principal
def main():

    file_name = '../resources/images/other.jpg'

    detector = ObjectDetector()

    # Detectar e classificar objetos na imagem
    image_with_classification, _, image_with_segmebtacion, _ = detector.process_image(file_name)

    Operations.display(image_with_classification, "Resultado")
    Operations.save_image(image_with_classification, '../resources/results/other/resultado.jpg')
    #Operations.display(image_with_segmebtacion, "Resultado")


if __name__ == "__main__":
    main()