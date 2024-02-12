from HistogramGenerator import HistogramAnalyzer
from EdgeDetection import EdgeDetector
from IdentifyingAndCountingObjects import ImageProcessor

def image_processor(file_name):
    processor = ImageProcessor(file_name)
    processor.process_and_show_results()

def generate_histogram_gray(file_name):
    analyzer = HistogramAnalyzer(file_name)
    analyzer.analyze_and_show_gray_histogram()

def generate_histogram_color(file_name):
    analyzer = HistogramAnalyzer(file_name)
    analyzer.analyze_and_show_color_histogram()

def edge_detection_canny(file_name):
    canny = EdgeDetector(file_name)
    canny.canny_detector()

def edge_detection_sobel(file_name):
    canny = EdgeDetector(file_name)
    canny.sobel_detector()

def edge_detection_laplacian(file_name):
    canny = EdgeDetector(file_name)
    canny.filter_laplacian()


# Função principal
def main():

    file_name = '../images/entrada.png'

    #image_processor(file_name)

    #test = ImageProcessor(file_name)
    #test.preprocess_image()



    # funçoes para verficar o histograma
    #generate_histogram_gray(file_name)
    generate_histogram_color(file_name)

    # funçoes para verficar a detecção de bordas
    #edge_detection_canny(file_name)
    #edge_detection_sobel(file_name)
    #edge_detection_laplacian(file_name)

    #imagem_borda = processamento(imagem_original)
    #limites(imagem_original, imagem_borda)


if __name__ == "__main__":
    main()