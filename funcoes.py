import numpy as np
import pandas as pd


def carregar_amostras(arquivo_excel):

    dados = pd.read_excel(arquivo_excel).to_numpy(dtype=int)

    X = dados[:-1]
    y = dados[-1]

    return X, y

def calcular_pesos(padroes):

    pesos = np.dot(padroes.T, padroes)

    np.fill_diagonal(pesos, 0)

    return pesos

def funcao_ativacao(valor):

    return 1 if valor >= 0 else -1

def atualizar_estado(estado, pesos):

    somas = np.dot(pesos, estado)

    novo_estado = np.where(somas >= 0, 1, -1)

    return novo_estado

def reconhecer_padrao(entrada, pesos, max_epocas=100):

    estado = entrada.copy()

    for epoca in range(1, max_epocas + 1):

        novo_estado = atualizar_estado(estado, pesos)

        if np.array_equal(novo_estado, estado):
            return novo_estado, epoca

        estado = novo_estado

    return estado, max_epocas


def encontrar_padrao_associado(estado, padroes):

    distancias = np.sum(padroes != estado, axis=1)

    indice = np.argmin(distancias)

    return indice, distancias[indice]