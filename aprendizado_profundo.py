import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re
from PIL import Image

def pre_processamento(diretorio, tamanho_maximo):
    # Converte as imagens em matrizes e escala de cinza
    img = Image.open(diretorio).convert('L')

    # Deixas a proporção das imagens em 1:1
    LARGURA, ALTURA = img.size
    if LARGURA != ALTURA:
            m_min_d = min(LARGURA, ALTURA)
            img = img.crop((0, 0, m_min_d, m_min_d))

    # Dimensiona as imagens em um tamanho padrão menor do que a resolução da imagem real
    img.thumbnail(tamanho_maximo, Image.Resampling.LANCZOS)

    return np.asarray(img)

def conjunto_dados(diretorio, tamanho_maximo):
        imagens = []
        nomes = []

        os.chdir(diretorio)
        for arquivo in glob.glob("*.png"):
            img = pre_processamento(arquivo, tamanho_maximo)
            if re.match('desmatamento.*', arquivo):
                    imagens.append(img)
                    nomes.append(0)
            elif re.match('nao_desmatada.*', arquivo):
                    imagens.append(img)
                    nomes.append(1)

        return (np.asarray(imagens), np.asarray(nomes))

tamanho_maximo = 130, 130
# Conjunto de dados para treinamento
(imagens_treinamento, nomes_treinamento) = conjunto_dados('./img/conjunto_dados/treinamento', tamanho_maximo)
# Conjunto de dados para testes
(imagens_testes, nomes_testes) = conjunto_dados('C:/Users/OI416936/github/deteccao-desmatamento/img/conjunto_dados/testes', tamanho_maximo)

nomes = ['desmatamento', 'floresta']

# 16 imagens de treinamento
imagens_treinamento.shape
(16, 130, 130)
print(nomes_treinamento)

# 6 imagens de testes
imagens_testes.shape
(6, 130, 130)
print(nomes_testes)