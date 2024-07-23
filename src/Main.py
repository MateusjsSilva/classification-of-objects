from util.OtherOperations import Operations
from util.IdentifyObjects import ObjectDetector
from util.HistogramOperations import HistogramProcessor as Hist

def main():

    file_name = './test/img/img2.jpg'

    # Detect and classify objects in the image
    image_with_classification, _, image_with_segmentacion, _ = ObjectDetector().process_image(file_name)

    Operations.display(image_with_classification, "Resultado")
    Operations.save_image(image_with_classification, './test/results/img2/resultado.jpg')

if __name__ == "__main__":
    main()