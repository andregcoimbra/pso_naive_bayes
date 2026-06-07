from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, precision_score,
                             roc_auc_score, roc_curve,
                             fbeta_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

from pyswarms.utils.plotters import plot_cost_history
import pyswarms as ps
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import sys

from rich.console import Console
from rich.table import Table
from rich import box

import warnings
warnings.filterwarnings('ignore')

def funcao_fitness(parametros, X_treino, y_treino, X_val, y_val, th_coluna, th_linha, penalizacao, expoente_penalizacao):
    """
    Avalia a qualidade de uma partícula do PSO.

    O vetor de parâmetros codifica:
      - posições [0 .. N_treino-1]              : relevância de cada instância de treino
      - posições [N_treino .. N_treino+N_feat-1] : relevância de cada feature
      - última posição                           : expoente do var_smoothing do GaussianNB (base 10)

    Retorna o valor de custo (quanto menor, melhor para o PSO).
    """
    # Extrai os sub-vetores que representam instâncias, colunas e suavização
    linhas_selecionadas  = parametros[:len(X_treino)]
    colunas_selecionadas = parametros[len(X_treino):(len(X_treino) + X_treino.shape[1])]
    suavizacao           = 10 ** parametros[-1]  # var_smoothing em escala logarítmica

    # Binariza a seleção com base nos limiares definidos
    colunas = colunas_selecionadas >= th_coluna
    linhas  = linhas_selecionadas  >= th_linha

    # Garante que ao menos uma feature seja mantida (evita conjunto vazio)
    if not np.any(colunas):
        colunas[np.argmax(colunas_selecionadas)] = True

    # Filtra features selecionadas nos conjuntos de treino e validação
    X_treino_selecionado = X_treino[:, colunas]
    X_val_selecionado    = X_val[:, colunas]

    # Filtra instâncias de treino selecionadas
    indices = np.where(linhas)[0]
    X_treino_selecionado = [X_treino_selecionado[i] for i in indices]
    y_treino_selecionado = [y_treino[i] for i in indices]

    # Descarta soluções em que apenas uma classe está presente no treino
    if len(set(y_treino_selecionado)) == 1:
        return 0

    # Treina o classificador Naive Bayes com as instâncias e features selecionadas
    modelo = GaussianNB(var_smoothing=suavizacao)
    modelo.fit(X_treino_selecionado, y_treino_selecionado)

    # Realiza predições no conjunto de validação
    y_previsto = modelo.predict(X_val_selecionado)

    # Calcula o F1-Score por classe e combina com pesos iguais (média macro manual)
    peso1   = .5
    peso2   = .5
    f1Score = f1_score(y_val, y_previsto, average=None)
    obj     = f1Score[0] * peso1 + f1Score[1] * peso2

    num_var_selecionadas = np.sum(colunas)

    # Função de custo: minimiza (-F1) penalizando o uso excessivo de features
    return -obj + penalizacao * (num_var_selecionadas ** expoente_penalizacao)

# ── Carregamento e preparação dos dados ──────────────────────────────────────

# Carrega os arquivos de treino e teste originais e os une em um único DataFrame
df = pd.read_csv('dados_chagas.csv')
df.set_index('Unnamed: 0', inplace=True)

colunas = df.columns[:-1]  # Nomes das features (todas exceto a coluna alvo)

X = df.iloc[:,:-1].values  # Matriz de features
y = df.iloc[:,-1].values   # Vetor de rótulos

# Divide em treino (80%) e teste (20%), preservando a proporção entre classes
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, shuffle=True, test_size=.2, stratify=y)

# Padroniza os dados (média=0, desvio=1); ajuste apenas no treino para evitar data leakage
scaler = StandardScaler()
X_treino_normalizado = scaler.fit_transform(X_treino)
X_teste_normalizado  = scaler.transform(X_teste)

# ── Configuração do PSO ──────────────────────────────────────────────────────

# Hiperparâmetros do PSO: w=inércia, c1=componente cognitiva, c2=componente social
parametros = {'w': 0.9, 'c1': 2, 'c2': 2}

# Estratégia de adaptação dos coeficientes ao longo das iterações
# Opções disponíveis: exp_decay | nonlin_mod | lin_variation
estrategia = {"w": "exp_decay", "c1": "lin_variation", "c2": "lin_variation"}

# Limiar para considerar uma feature/instância como selecionada
th_coluna = 0.7
th_linha  = 0.5

# Penalização proporcional ao número de features selecionadas
penalizacao          = (1 / X_treino.shape[1])
expoente_penalizacao = 2

# Espaço de busca: instâncias e features em [0,1]; expoente de suavização em [-9, 0]
limite_superior = [1] * (len(X_treino) + X_treino.shape[1]) + [0]
limite_inferior = [0] * (len(X_treino) + X_treino.shape[1]) + [-9]
velocidade = (-.5, .5)  # Clamp de velocidade (mínima, máxima)
n_particulas = 50
epocas       = 200

def funcao_objetivo(parametros):
    """Wrapper vetorizado exigido pelo PySwarms: avalia todas as partículas do enxame."""
    n_particulas = parametros.shape[0]
    obj = np.zeros(n_particulas)
    for i in range(n_particulas):
        obj[i] = funcao_fitness(
            parametros[i], X_treino_normalizado, y_treino,
            X_teste_normalizado, y_teste,
            th_coluna, th_linha, penalizacao, expoente_penalizacao
        )
    return obj

# ── Execução do PSO ───────────────────────────────────────────────────────────

# Posição inicial das partículas distribuída uniformemente no espaço de busca
init_pos = np.random.uniform(low=limite_inferior, high=limite_superior, size=(n_particulas, len(limite_superior)))

# Instancia o otimizador PSO com topologia global (cada partícula conhece a melhor global)
modelo = ps.single.GlobalBestPSO(
    n_particles=n_particulas,
    dimensions=len(limite_superior),
    options=parametros,
    oh_strategy=estrategia,
    bounds=(limite_inferior, limite_superior),
    velocity_clamp=velocidade,
    init_pos=init_pos
)

# Executa a otimização e retorna o melhor custo e as variáveis correspondentes
objetivo, variaveis = modelo.optimize(funcao_objetivo, iters=epocas, verbose=True)

# Inverte o sinal do histórico de custo (PSO minimiza -F1, então invertemos para exibir F1 crescente)
dados_grafico = [-x for x in modelo.cost_history]
grafico = plot_cost_history(cost_history=dados_grafico, title='Histórico')
plt.show()  # Exibe a janela do gráfico


# ── Reconstrução da melhor solução encontrada ────────────────────────────────

# Binariza os vetores da melhor partícula usando os mesmos limiares do fitness
linhas_selecionadas  = variaveis[:len(X_treino)] > th_linha
colunas_selecionadas = variaveis[len(X_treino):(len(X_treino) + X_treino.shape[1])] > th_coluna

# Garante ao menos uma feature selecionada
if not np.any(colunas_selecionadas):
    colunas_selecionadas[np.argmax(colunas_selecionadas)] = True

# Aplica a seleção de features
X_treino_selecionado = X_treino_normalizado[:, colunas_selecionadas]
X_teste_selecionado  = X_teste_normalizado[:, colunas_selecionadas]

# Aplica a seleção de instâncias de treino
indices = np.where(linhas_selecionadas)[0]
X_treino_selecionado = [X_treino_selecionado[i] for i in indices]
y_treino_selecionado = [y_treino[i] for i in indices]

# Treina o modelo final com os subconjuntos selecionados pelo PSO
modelo = GaussianNB(var_smoothing=10 ** variaveis[-1])
modelo.fit(X_treino_selecionado, y_treino_selecionado)

# ── Avaliação do modelo final ─────────────────────────────────────────────────

# Obtém predições e calcula métricas por classe
y_previsto        = modelo.predict(X_teste_selecionado)
acc               = accuracy_score(y_teste, y_previsto)
var_f1_score      = f1_score(y_teste, y_previsto, average=None)
var_recall        = recall_score(y_teste, y_previsto, average=None)
var_precision     = precision_score(y_teste, y_previsto, average=None)
# F2-Score: pondera recall duas vezes mais que precisão (foco em falsos negativos)
var_f2_score      = fbeta_score(y_teste, y_previsto, beta=2, average=None)[1]  # F2 para a classe positiva
var_roc_auc_score = roc_auc_score(y_teste, y_previsto)

# Calcula a curva ROC usando as probabilidades estimadas para a classe positiva
y_probabilidades = modelo.predict_proba(X_teste_selecionado)[:, 1]
fpr, tpr, _ = roc_curve(y_teste, y_probabilidades)

# ── Registro dos resultados ───────────────────────────────────────────────────

# Consolida todas as métricas em um DataFrame para exportação
data = {
    'Accuracy': [round(acc * 100, 2)],
    'Recall_Class1': [round(var_recall[1] * 100, 2)],
    'Precision_Class1': [round(var_precision[1] * 100, 2)],
    'F1_Score_Class1': [round(var_f1_score[1] * 100, 2)],
    'F2_Score_Class1': [round(var_f2_score * 100, 2)],
    'Recall_Class0': [round(var_recall[0] * 100, 2)],
    'Precision_Class0': [round(var_precision[0] * 100, 2)],
    'F1_Score_Class0': [round(var_f1_score[0] * 100, 2)],
    'FPR': [fpr],
    'TPR': [tpr],
    'ROC_AUC': [round(var_roc_auc_score * 100, 2)],
    'Variables': [list(colunas_selecionadas).count(True)],       # Nº de features selecionadas
    'Instances_train': [list(linhas_selecionadas).count(True)]  # Nº de instâncias de treino usadas
}

# Exibe resumo tabular no console com Rich
console = Console()

tabela = Table(title="Resultado PSO + Naive Bayes", box=box.ROUNDED, show_lines=True)

tabela.add_column("Métrica",        style="bold cyan",   justify="left")
tabela.add_column("Classe 1 (+)",   style="bold green",  justify="right")
tabela.add_column("Classe 0 (−)",   style="bold yellow", justify="right")

tabela.add_row("Acurácia",   f"{round(acc * 100, 2):.2f}%",                        "")
tabela.add_row("Recall",     f"{round(var_recall[1] * 100, 2):.2f}%",              f"{round(var_recall[0] * 100, 2):.2f}%")
tabela.add_row("Precisão",   f"{round(var_precision[1] * 100, 2):.2f}%",           f"{round(var_precision[0] * 100, 2):.2f}%")
tabela.add_row("F1-Score",   f"{round(var_f1_score[1] * 100, 2):.2f}%",           f"{round(var_f1_score[0] * 100, 2):.2f}%")
tabela.add_row("F2-Score",   f"{round(var_f2_score * 100, 2):.2f}%",              "")
tabela.add_row("ROC AUC",    f"{round(var_roc_auc_score * 100, 2):.2f}%",         "")
tabela.add_row("Features",   str(list(colunas_selecionadas).count(True)),          "")
tabela.add_row("Instâncias", str(list(linhas_selecionadas).count(True)),           "")

features_selecionadas = list(colunas[colunas_selecionadas])
tabela.add_row("Features selecionadas", ", ".join(features_selecionadas), "")

console.print(tabela)

