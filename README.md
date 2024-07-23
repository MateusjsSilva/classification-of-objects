# Classification of objects (PDI)
Este repositório contém um projeto para classificação de objetos em imagens, distinguindo-os por formato e cor. O código implementa uma classe ObjectDetector que processa imagens para detectar, classificar e contar objetos com base em suas formas e cores. O processo inclui a remoção de sombras, segmentação da imagem, detecção de contornos e classificação dos objetos por forma (triângulo, quadrado, retângulo, círculo) e cor. Note: para cada imagem é necessario ajustar os parametros para melhor identificação.

## Captura de tela
<div align="center">
  <img src="test/results/img2/resultado.jpg" height="400em">
</div>

## Como executar

1. Certifique-se de que todas as dependências estejam instaladas:
  ```sh
  pip install -r .\src\requeriments.txt
  ```

2. Execute o script:
  ```sh
  python .\src\main.py
  ```

## Contribuição

Sinta-se à vontade para abrir issues ou enviar pull requests. Toda contribuição é bem-vinda!

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.