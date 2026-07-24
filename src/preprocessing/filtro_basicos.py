import cv2
import numpy as np

# função que ordena os pontos do contorno da folha; (x,y) -> x = horizontal, y = vertical;
def ordenar_pontos(pts):
    pts = pts.reshape(4, 2)
    ret = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1) # pega os dois extremos (ponto superior esquerdo e inferior direito); x+y
    ret[0] = pts[np.argmin(s)] 
    ret[2] = pts[np.argmax(s)] 
    
    diff = np.diff(pts, axis=1) # pega os outros dois extremos (ponto superior direito e inferior esquerdo); y-x
    ret[1] = pts[np.argmin(diff)] # um y muito pequeno e x muito grande (superior direito)
    ret[3] = pts[np.argmax(diff)] # um y muito grande e x muito pequeno (inferior esquerdo)
    
    return ret

def processar_imagem_completo(caminho_imagem):
    img_original = cv2.imread(caminho_imagem) # carrega a imagem do disco
    if img_original is None: # se o caminho for inválido ou a imagem não puder ser carregada, exibe uma mensagem de erro e retorna
        print("Erro ao carregar a imagem.")
        return

    # acelerar a detecção de contornos redimensionando a imagem para 800px de altura, mantendo a proporção da largura

    proporcao = 800.0 / img_original.shape[0] #img_original.shape[0] é a altura da imagem original; proporcao é a razão entre a altura desejada (800px) e a altura original
    dimensoes = (int(img_original.shape[1] * proporcao), 800) #img_original.shape[1] é a largura da imagem original; dimensoes é uma tupla com a largura e altura desejadas para redimensionar a imagem
    img_redimensionada = cv2.resize(img_original, dimensoes, interpolation=cv2.INTER_AREA)# redimensiona a imagem original para 800px de altura, mantendo a proporção da largura

    # escurecer a imagem, aplicar desfoque e detectar bordas para encontrar contornos

    img_cinza = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2GRAY)
    img_desfoque = cv2.GaussianBlur(img_cinza, (5, 5), 0) #desfoque gaussiano para reduzir ruído e detalhes na imagem, facilitando a detecção de bordas
    img_bordas = cv2.Canny(img_desfoque, 75, 200) 

    # encontrar contornos a partir das bordas detectadas e ordenar pelo tamanho (maior primeiro)

    contornos, _ = cv2.findContours(img_bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True) #garantir que o maior contorno seja o primeiro da lista, que provavelmente será a folha

    if len(contornos) == 0:
        print("Não foi possível encontrar nenhum objeto na imagem.")
        cv2.imshow("Deteccao e Geometria", img_redimensionada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    maior_contorno = contornos[0]
    hull = cv2.convexHull(maior_contorno) # cria um contorno convexo a partir do maior contorno encontrado, que é uma forma simplificada do contorno original, eliminando concavidades e irregularidades
    perimetro = cv2.arcLength(hull, True) # calcula o perímetro do contorno convexo, que é a soma das distâncias entre os pontos do contorno 

    contorno_folha = None
    for multiplicador in [0.02, 0.03, 0.04, 0.05, 0.06]:
        aproximacao = cv2.approxPolyDP(hull, multiplicador * perimetro, True) # aproxima o contorno convexo para um polígono com menos vértices, controlando a precisão da aproximação com o multiplicador do perímetro
        if len(aproximacao) == 4:
            contorno_folha = aproximacao
            break

    if contorno_folha is None:
        print("O contorno base não conseguiu formar 4 lados.")
        cv2.imshow("Deteccao e Geometria", img_redimensionada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    pontos_ordenados = ordenar_pontos(contorno_folha)

    # pegar a imagem original e aplicar a transformação de perspectiva para obter uma visão "de cima" da folha 

    fator_inverso = 1.0 / proporcao
    pontos_originais = (pontos_ordenados * fator_inverso).astype("float32")
    (tl, tr, br, bl) = pontos_originais

    # calcular a largura e altura máximas da folha para definir o tamanho da imagem transformada
    # usa a fórmula da distância euclidiana para calcular a largura e altura da folha a partir dos pontos ordenados

    larguraA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    larguraB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    largura_maxima = max(int(larguraA), int(larguraB))

    alturaA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    alturaB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    altura_maxima = max(int(alturaA), int(alturaB))

    # definir os pontos de destino para a transformação de perspectiva, que serão os cantos da imagem final retangular
    dst = np.array([
        [0, 0],
        [largura_maxima - 1, 0],
        [largura_maxima - 1, altura_maxima - 1],
        [0, altura_maxima - 1]
    ], dtype="float32")

    # calcular a matriz de transformação de perspectiva e aplicar a transformação na imagem original para obter a folha alinhada

    matriz_transformacao = cv2.getPerspectiveTransform(pontos_originais, dst)
    img_escaneada = cv2.warpPerspective(img_original, matriz_transformacao, (largura_maxima, altura_maxima))

    # aplicar um desfoque suave na imagem escaneada para reduzir ruído e melhorar a nitidez, e depois combinar a imagem original com o desfoque para obter uma imagem mais nítida
    
    desfoque_suave = cv2.GaussianBlur(img_escaneada, (0, 0), sigmaX=3)
    img_nitida = cv2.addWeighted(img_escaneada, 1.5, desfoque_suave, -0.5, 0)

    img_escaneada_cinza = cv2.cvtColor(img_nitida, cv2.COLOR_BGR2GRAY)

    img_binarizada = cv2.adaptiveThreshold( # aplica a binarização adaptativa na imagem em tons de cinza para separar o texto do fundo, facilitando a leitura e o processamento OCR
        img_escaneada_cinza,
        255, # valor máximo para os pixels binarizados (branco)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25, # tamanho do bloco (área local) para calcular o limiar adaptativo;
        15 # valor subtraído do limiar calculado para ajustar a binarização 
    )

   
    cv2.imwrite("data/processed/prova_binarizada.jpg", img_binarizada)
    cv2.imshow("Folha Binarizada (Pronta para OCR)", img_binarizada)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    processar_imagem_completo("data/raw/teste.jpg")
