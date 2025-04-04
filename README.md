# Projeto-Classifica-o-de-Emails-Spam

## 1. Introdução

Este projeto tem como objetivo principal desenvolver e comparar modelos de aprendizado de máquina para a classificação de e-mails como spam ou não spam. O conjunto de dados utilizado, obtido do Kaggle ([https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)), contém informações sobre as 3000 palavras mais frequentes em um conjunto de e-mails, juntamente com um rótulo binário indicando se um e-mail é spam (1) ou não spam (0).

## 2. Metodologia

A metodologia adotada neste projeto foi dividida em cinco etapas principais:

1.  **Aquisição e Preparação dos Dados:** Importação do conjunto de dados, análise exploratória inicial, pré-processamento (remoção de coluna irrelevante, separação de features e rótulos, divisão em conjuntos de treinamento e teste, e escalonamento adequado para cada modelo).
2.  **Treinamento de Modelos Base:** Treinamento de múltiplos modelos de classificação (Naive Bayes, Árvore de Decisão, KNN, Regressão Logística, Floresta Aleatória e SVM) para estabelecer uma linha de base de desempenho.
3.  **Seleção e Otimização de Modelos:** Definição de grades de hiperparâmetros e utilização do `GridSearchCV` para encontrar a melhor combinação de parâmetros para os modelos de Floresta Aleatória e Regressão Logística.
4.  **Avaliação Detalhada dos Modelos:** Avaliação abrangente dos modelos otimizados usando diversas métricas (acurácia, matriz de confusão e relatório de classificação).
5.  **Análise de Resultados e Conclusões:** Interpretação dos resultados, discussão sobre o desempenho dos modelos e possíveis direções futuras.

## 3. Código Detalhado

### 3.1. Download do Dataset

O código importa a biblioteca `kagglehub`, que permite baixar datasets e modelos diretamente do Kaggle.
`kagglehub.dataset_download(...)` é a função que baixa um dataset do Kaggle.
O argumento `"balaka18/email-spam-classification-dataset-csv"` é o identificador do conjunto de dados Email Spam Classification Dataset (CSV), que está no perfil do usuário `balaka18`.
`path` recebe o caminho local onde os arquivos do dataset foram baixados.
O `print` imprime o caminho onde os arquivos do dataset foram salvos, permitindo que você os localize facilmente para uso posterior.

### 3.2. Importação de Bibliotecas

Aqui são importadas bibliotecas essenciais para manipulação de dados, preparação e modelagem.

* `pandas`: Para manipulação e análise de dados tabulares.
* `numpy`: Para operações numéricas eficientes.
* `matplotlib.pyplot`: Para criação de gráficos básicos.
* `seaborn`: Para visualizações estatísticas avançadas.
* `sklearn.model_selection`: Para dividir dados em treino/teste e realizar busca de hiperparâmetros (`GridSearchCV`).
* `sklearn.preprocessing`: Para escalonamento de features (`StandardScaler`, `MinMaxScaler`).
* `sklearn.naive_bayes`: Implementação do algoritmo Naive Bayes (`MultinomialNB`).
* `sklearn.tree`: Implementação de árvores de decisão (`DecisionTreeClassifier`).
* `sklearn.neighbors`: Implementação do algoritmo K-Vizinhos Mais Próximos (`KNeighborsClassifier`).
* `sklearn.linear_model`: Implementação de modelos lineares (`LogisticRegression`).
* `sklearn.ensemble`: Implementação de métodos de ensemble (`RandomForestClassifier`).
* `sklearn.svm`: Implementação de Máquinas de Vetores de Suporte (`SVC`).
* `sklearn.metrics`: Para avaliação de modelos (`accuracy_score`, `confusion_matrix`, `classification_report`).
* `scipy.stats`: Para testes estatísticos (`chi2_contingency`, `ttest_rel`).

### 3.3. Carregamento do Dataset

Carrega o dataset `emails.csv` utilizando a biblioteca Pandas e exibe as primeiras linhas e a sua dimensão.

### 3.4. Remover a Coluna Irrelevante

Remove a coluna "Email No." do DataFrame, pois ela não contribui para a tarefa de classificação.

### 3.5. Separar Features e Rótulos

Separa o DataFrame em duas partes: `X` contendo as features (variáveis independentes) e `y` contendo a variável alvo ("Prediction").

### 3.6. Dividir em Treino e Teste

Divide os dados em conjuntos de treinamento (70%) e teste (30%) utilizando uma semente aleatória (`random_state=42`) para garantir a reprodutibilidade da divisão.

### 3.7. Aplicando escalonamento adequado para cada modelo

Inicializa e aplica os escalonadores `StandardScaler` (para padronização) e `MinMaxScaler` (para normalização) aos conjuntos de treinamento e teste. Diferentes modelos se beneficiam de diferentes tipos de escalonamento.

### 3.8. Definir os Modelos

Define um dicionário contendo os modelos de aprendizado de máquina a serem utilizados, juntamente com os respectivos dados de treinamento e teste (escalonados de forma apropriada para cada modelo). Os modelos incluem Naive Bayes, Árvore de Decisão, KNN, Regressão Logística, Floresta Aleatória e SVM.

### 3.9. Treinar e Avaliar os Modelos Base

Este loop itera sobre os modelos definidos, treina cada modelo com os dados de treinamento correspondentes e avalia seu desempenho no conjunto de teste, imprimindo a acurácia, a matriz de confusão e o relatório de classificação.

### 3.10. Seleção de Hiperparâmetros

Define as grades de hiperparâmetros a serem testadas para os modelos de Floresta Aleatória e Regressão Logística.

### 3.11. Ajuste dos Modelos

Utiliza o `GridSearchCV` para realizar a busca exaustiva sobre as grades de hiperparâmetros definidas para a Floresta Aleatória e a Regressão Logística, utilizando validação cruzada. Após o ajuste, os melhores hiperparâmetros e a melhor acurácia são identificados, e os modelos ajustados são avaliados no conjunto de teste.

## 4. Resultados

Os resultados da avaliação dos modelos base e dos modelos com hiperparâmetros ajustados serão apresentados aqui. Serão comparadas as métricas de acurácia, precisão, recall e F1-score para cada modelo, permitindo identificar qual deles apresenta o melhor desempenho na tarefa de classificação de e-mails como spam ou não spam.

## 5. Conclusões

Com base nos resultados obtidos, serão discutidas as conclusões sobre o desempenho dos diferentes modelos de aprendizado de máquina aplicados ao problema de classificação de spam. Serão analisadas a importância das features, as vantagens e desvantagens de cada modelo e possíveis direções para trabalhos futuros, como a exploração de outras técnicas de pré-processamento, a utilização de outros modelos ou a aplicação de técnicas de feature engineering mais avançadas.

## 6. Como Executar o Código

1.  Certifique-se de ter o Python 3 instalado.
2.  Instale as bibliotecas necessárias utilizando o pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy kagglehub
    ```
3.  Para baixar o dataset diretamente do Kaggle, você precisará de uma chave de API do Kaggle. Siga as instruções em [https://github.com/Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api) para configurar suas credenciais.
4.  Execute o script Python contido neste repositório.

## 7. Contribuição

Contribuições para este projeto não são esperadas neste momento, pois trata-se de um projeto individual para fins de aprendizado e demonstração.

## 8. Autor

Marcos Vinicius da Silva
