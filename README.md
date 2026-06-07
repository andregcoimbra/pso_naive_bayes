# PSO + Naive Bayes — Classificação Doença de Chagas

Otimização por Enxame de Partículas (PSO) aplicada à seleção simultânea de **features** e **instâncias de treino** para um classificador Gaussian Naive Bayes na base de dados da Doença de Chagas.

---

## Descrição

O algoritmo PSO busca, em conjunto:

- Quais **features** (colunas) do conjunto de dados devem ser utilizadas.
- Quais **instâncias** (linhas) do conjunto de treino devem ser incluídas.
- O valor ideal do hiperparâmetro `var_smoothing` do GaussianNB (em escala logarítmica).

A função de fitness maximiza o F1-Score médio entre as duas classes, penalizando soluções com muitas features selecionadas para favorecer modelos mais simples.

---

## Estrutura do projeto

```
pso_naive_base_chagas/
├── pso_naivebayes.py   # Script principal
├── dados_chagas.csv    # Base de dados (Doença de Chagas)
├── requirements.txt    # Dependências Python
└── README.md           # Este arquivo
```

---

## Instalação

```bash
pip install -r requirements.txt
```

> Recomenda-se o uso de um ambiente virtual (`venv` ou `conda`) com **Python 3.11**.

---

## Uso

```bash
python pso_naivebayes.py
```

Ao final da execução, o console exibe uma tabela com as seguintes métricas avaliadas no conjunto de teste:

| Métrica               | Descrição                                          |
|-----------------------|----------------------------------------------------|
| Acurácia              | Proporção de acertos geral                         |
| Recall                | Sensibilidade por classe                           |
| Precisão              | Precisão por classe                                |
| F1-Score              | Média harmônica Precisão/Recall por classe         |
| F2-Score              | F-beta (β=2) da classe positiva                    |
| ROC AUC               | Área sob a curva ROC                               |
| Features              | Número de features selecionadas pelo PSO           |
| Instâncias            | Número de instâncias de treino selecionadas        |
| Features selecionadas | Nomes das features escolhidas pelo PSO             |

Além disso, é exibido o gráfico do histórico do F1-Score ao longo das épocas do PSO.

---

## Hiperparâmetros do PSO

| Parâmetro               | Valor                          |
|-------------------------|--------------------------------|
| Nº de partículas        | 50                             |
| Épocas                  | 200                            |
| Inércia (`w`)           | 0.9 (decaimento exponencial)   |
| Coef. cognitivo (`c1`)  | 2.0 (variação linear)          |
| Coef. social (`c2`)     | 2.0 (variação linear)          |
| Limiar features         | 0.7                            |
| Limiar instâncias       | 0.5                            |
| Penalização             | 1 / nº_features                |
| Expoente penalização    | 2                              |

---

## Dependências

Veja [requirements.txt](requirements.txt).

---

## Origem dos dados

A base de dados utilizada neste projeto tem origem no conjunto de dados do estudo SaMi-Trop:

> SaMi-Trop. **Chagas disease dataset**. 2022. Acesso em: 7 jul. 2024.
> Disponível em: <http://journals.plos.org/plosntds/article/asset?unique&id=info:doi/10.1371/journal.pntd.0010356.s003>
