# Implementação Rede Hopfield

Este projeto implementa o algoritmo da Rede Neural de Hopfield em Python para armazenamento e reconhecimento de padrões.

---

### 🛠️ Funcionalidades:

- **Inicialização / Cálculo dos pesos:**

  - Realiza a leitura de um arquivo `.xlsx` com as amostras disponíveis para treinar a rede
  - Utiliza as amostras conhecidas para calcular a matriz de pesos da rede
  - Define os pesos das conexões entre os neurônios
  - Mostra o resultado do treinamento, ou seja, a matriz de pesos obtida

- **Teste / Inferência:**

  - Utiliza a última linha do arquivo `.xlsx` como padrão de entrada desconhecido
  - Realiza o processo iterativo de reconhecimento do padrão
  - Mostra em tela o padrão ao qual a entrada está associada
  - Mostra a quantidade de épocas necessárias para a estabilização da rede
  - Mostra o erro cometido durante o reconhecimento

---

### ⚙️ Pré-requisitos:

- **Linguagem:** Python 3.12+
- **Bibliotecas:**

  - `numpy` - Operações matriciais e vetoriais
  - `pandas` - Leitura e manipulação dos dados
  - `openpyxl` - Leitura de arquivos `.xlsx`

---

### 👾 Como executar:

- **Local:**
    <br>Com o Python 3.12+ e as bibliotecas necessárias instaladas, abra o VS Code (ou sua IDE de preferência) e execute:
    - `git clone https://github.com/gabrielpassos7/modelo-hopfield` - Para clonar o repositório para sua máquina
    - `python hopfield.py` - Para inicializar a rede e apresentar os pesos obtidos
    - `python testar_modelo.py` - Para testar a rede com o padrão desconhecido presente no arquivo Excel

- **Codespaces:**
    <br>No repositório do GitHub, clique em ![](https://img.shields.io/badge/%3C%3E_Code_▾-238636?style=flat), depois em `Codespaces` e por fim em ![](https://img.shields.io/badge/Create_codespace_on_main-238636?style=flat)
    <br>Assim que o Codespace for criado, execute no terminal:
    - `pip install numpy pandas openpyxl` - Para instalar as bibliotecas necessárias
    - `python hopfield.py` - Para realizar inicializar a rede e apresentar os pesos obtidos
    - `python testar_modelo.py` - Para testar a rede com o padrão desconhecido presente no arquivo Excel

---

### 🧮 Fundamentação matemática e implementação:

- **Cálculo dos pesos:**

A matriz de pesos da Rede Hopfield é obtida a partir dos padrões que serão armazenados.

$$w_{ij} = \sum_{r=1}^{M} x_i^r x_j^r \text{~~~~,~~~~se~~} i \neq j$$

$$w_{ij} = 0 \text{~~~~,~~~~se~~} i = j$$

Onde $w_{ij}$ representa o peso da conexão entre os neurônios $i$ e $j$, $x_i^r$ representa o i-ésimo elemento do padrão $r$ e $M$ representa a quantidade de padrões armazenados.

<br>

```
def calcular_pesos(padroes):

    pesos = np.dot(padroes.T, padroes)

    np.fill_diagonal(pesos, 0)

    return pesos
```

<br>

- **Somatória dos pesos:**

Durante o processo iterativo, cada neurônio recebe uma soma ponderada das saídas dos demais neurônios:

$$u_j(t) = \sum_i w_{ij}y_i(t)$$

Onde $u_j(t)$ representa a entrada do neurônio $j$ no instante $t$, $w_{ij}$ representa os pesos das conexões e $y_i(t)$ representa a saída do neurônio $i$ no instante $t$.

<br>

```
somas = np.dot(pesos, estado)
```

<br>

- **Nova saída e Função de ativação:**

A nova saída do neurônio é obtida aplicando a função de ativação sobre o valor calculado:

$$y_j(t+1) = f_j[u_j(t)]$$

Nesta implementação, é utilizada uma função de ativação bipolar, na qual os neurônios podem assumir os valores `+1` ou `-1`:

$$
f(u) =
\begin{cases}
+1, & \text{se } u \geq 0 \\
-1, & \text{se } u < 0
\end{cases}
$$

<br>

```
novo_estado = np.where(somas >= 0, 1, -1)
```

<br>

- **Processo iterativo:**

A rede recebe um padrão desconhecido como estado inicial e realiza sucessivas atualizações até atingir um estado de estabilização.

$$y(t+1) = f(Wy(t))$$

Onde $W$ representa a matriz de pesos e $y(t)$ representa o estado atual da rede.

<br>

```
for epoca in range(1, max_epocas + 1):

    novo_estado = atualizar_estado(estado, pesos)

    if np.array_equal(novo_estado, estado):
        return novo_estado, epoca

    estado = novo_estado
```

<br>

- **Reconhecimento do padrão:**

Após a estabilização, o estado final da rede é comparado com os padrões armazenados para identificar o padrão ao qual a entrada está mais associada.

O erro é calculado através da quantidade de componentes diferentes entre o padrão recuperado e cada padrão armazenado:

$$erro = \sum_i [y_i \neq x_i]$$

Onde $y_i$ representa o estado final da rede e $x_i$ representa o i-ésimo elemento do padrão armazenado.

<br>

```
distancias = np.sum(padroes != estado, axis=1)

indice = np.argmin(distancias)

return indice, distancias[indice]
```

<br>

---

### 📄 Referências:
