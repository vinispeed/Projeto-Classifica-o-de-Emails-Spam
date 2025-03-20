# Projeto-Classifica-o-de-Emails-Spam

1. Introdução
Este projeto tem como objetivo principal desenvolver e comparar modelos de aprendizado de máquina para a classificação de e-mails como spam ou não spam. O conjunto de dados utilizado, obtido do Kaggle, contém informações sobre as 3000 palavras mais frequentes em um conjunto de e-mails, juntamente com um rótulo binário indicando se um e-mail é spam (1) ou não spam (0).
2. Metodologia
A metodologia adotada neste projeto foi dividida em cinco etapas principais:
Aquisição e Preparação dos Dados: Importação do conjunto de dados, análise exploratória inicial, pré-processamento e divisão em conjuntos de treinamento e teste.
Treinamento de Modelos Base: Treinamento de vários modelos de classificação para estabelecer uma linha de base de desempenho.
Seleção e Otimização de Modelos: Seleção dos modelos com melhor desempenho na etapa anterior e otimização de seus hiperparâmetros.
Avaliação Detalhada dos Modelos: Avaliação abrangente dos modelos otimizados usando diversas métricas e testes estatísticos.
Análise de Resultados e Conclusões: Interpretação dos resultados, discussão sobre a importância das características e conclusões sobre o desempenho dos modelos.
3. Código Detalhado
3.1. Download do Dataset
Python
import kagglehub

path = kagglehub.dataset_download("balaka18/email-spam-classification-dataset-csv")

print("Path to dataset files:", path)

O código importa a biblioteca kagglehub, que permite baixar datasets e modelos diretamente do Kaggle.
kagglehub.dataset_download(...) é a função que baixa um dataset do Kaggle.
O argumento "balaka18/email-spam-classification-dataset-csv" é o identificador do conjunto de dados Email Spam Classification Dataset (CSV), que está no perfil do usuário balaka18.
path recebe o caminho local onde os arquivos do dataset foram baixados.
O print imprime o caminho onde os arquivos do dataset foram salvos, permitindo que você os localize facilmente para uso posterior.
3.2. Importação de Bibliotecas
Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import chi2_contingency, ttest_rel
from scipy import stats

Aqui são importadas bibliotecas essenciais para manipulação de dados, preparação e modelagem.
pandas é essencial para carregar e manipular conjuntos de dados tabulares (como arquivos CSV).
numpy é usado para manipulação de arrays e operações matemáticas avançadas.
matplotlib.pyplot permite criar gráficos como histogramas, scatter plots, etc.
seaborn facilita a criação de gráficos estatísticos com layouts mais bonitos.
train_test_split: divide os dados em conjunto de treino e teste.
GridSearchCV: faz ajuste de hiperparâmetros testando várias combinações para encontrar a melhor configuração do modelo.
StandardScaler: padroniza os dados transformando-os para uma distribuição normal (média 0 e desvio padrão 1).
MinMaxScaler: normaliza os dados para ficarem no intervalo [0,1].
Multinomial Naive Bayes (MultinomialNB): Algoritmo probabilístico baseado no Teorema de Bayes, usado para classificação de textos como detecção de spam.
Árvore de Decisão (DecisionTreeClassifier): Modelo que toma decisões com base em regras Sim/Não, aplicado em diagnóstico médico e análise de crédito.
K-Nearest Neighbors (KNeighborsClassifier): Classifica novos dados comparando com os K vizinhos mais próximos, usado em reconhecimento facial e sistemas de recomendação.
Regressão Logística (LogisticRegression): Algoritmo de classificação binária baseado em probabilidades, aplicado em detecção de fraudes e diagnóstico médico.
Random Forest (RandomForestClassifier): Conjunto de múltiplas árvores de decisão para maior precisão, utilizado em classificação de imagens e previsão de preços.
Support Vector Machine (SVC): Algoritmo que encontra o hiperplano ótimo para separar classes, usado em classificação de texto e diagnóstico médico avançado.
accuracy_score: mede a acurácia do modelo.
confusion_matrix: mostra os erros e acertos do modelo (falsos positivos, falsos negativos, etc.).
classification_report: exibe métricas como precisão, recall e F1-score.
chi2_contingency: realiza o teste qui-quadrado, usado para avaliar a independência entre variáveis categóricas.
3.3. Carregamento do Dataset
Python
file_path = "/root/.cache/kagglehub/datasets/balaka18/email-spam-classification-dataset-csv/versions/1/emails.csv"
df = pd.read_csv(file_path)
df.head()
df.shape

Aqui, os dados são carregados a partir de um arquivo CSV.
file_path contém o caminho do dataset, que está armazenado no ambiente do Kaggle.
pd.read_csv(file_path) carrega o CSV em um DataFrame Pandas (df).
df.head() exibe as 5 primeiras linhas do dataset, permitindo verificar sua estrutura.
Esse dataset contém e-mails rotulados como spam ou não spam, e o objetivo é treinar os modelos de classificação para prever esses rótulos.
df.shape retorna uma tupla no formato (n_linhas, n_colunas), onde:
n_linhas → Número total de amostras (e-mails no dataset).
n_colunas → Quantidade de atributos (como texto do e-mail, rótulo spam/não spam, etc.).
3.4. Remover a Coluna Irrelevante
Python
df.drop(columns=["Email No."], inplace=True)

A função drop() é usada para remover uma ou mais colunas ou linhas de um DataFrame Pandas.
columns=["Email No."]: Especifica que queremos remover a coluna chamada "Email No.".
inplace=True: Garante que a operação seja feita diretamente no DataFrame original (df). Ou seja, a coluna será removida do df sem precisar criar um novo DataFrame. Se fosse inplace=False (o padrão), a operação retornaria um novo DataFrame e o original permaneceria intacto.
A coluna "Email No." contém um nome fictício para cada e-mail, o que não tem relevância para a tarefa de classificação (onde queremos focar em atributos como o conteúdo do e-mail e seu rótulo - spam ou não).
3.5. Separar Features e Rótulos
Python
X = df.drop(columns=["Prediction"])  # Todas as colunas, exceto "Prediction"
y = df["Prediction"]  # Coluna-alvo

Estamos criando a variável X, que irá armazenar todas as colunas do DataFrame df, exceto a coluna "Prediction", que é a variável alvo.
df.drop(columns=["Prediction"]): O método drop() é utilizado para remover a coluna "Prediction" do DataFrame df. Criando um novo DataFrame sem essa coluna, e o resultado é armazenado em X.
X: Representa as features (características) do dataset, ou seja, são os dados que o modelo utilizará para fazer previsões.
Estamos também criando a variável y, que irá armazenar a coluna alvo ("Prediction").
df["Prediction"]: Estamos acessando a coluna "Prediction" no DataFrame df, que contém os rótulos dos e-mails (por exemplo, Spam ou Não Spam).
y: Representa a variável alvo (target) do modelo, ou seja, é o valor que queremos prever com base nas features. No caso, o objetivo é prever se o e-mail é Spam ou Não Spam.
3.6. Dividir em Treino e Teste
Python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
A função train_test_split() do scikit-learn é utilizada para dividir os dados em dois conjuntos: um para treinamento e outro para teste. Ela recebe várias opções de parâmetros:
Parâmetros:
X: São as features (variáveis independentes), ou seja, os dados de entrada que o modelo usará para aprender.
y: A variável alvo (target), que é o que o modelo tentará prever (no caso, a classificação dos e-mails como Spam ou Não Spam).
Parâmetros adicionais:
test_size=0.3: Especifica a proporção dos dados a ser utilizada para o conjunto de teste. Nesse caso, 0.3 significa que 30% dos dados serão usados para teste, e os 70% restantes serão usados para treinamento.
random_state=42: Define uma semente aleatória para garantir que a divisão dos dados seja reproduzível. Ou seja, toda vez que você executar o código com random_state=42, a divisão dos dados será a mesma, o que é útil para garantir resultados consistentes e comparáveis.
Retorno:
A função train_test_split() retorna quatro objetos:
X_train: Conjunto de dados de entrada (features) para o treinamento do modelo.
X_test: Conjunto de dados de entrada para avaliação do modelo.
y_train: Rótulos (variável alvo) correspondentes ao conjunto de treinamento.
y_test: Rótulos (variável alvo) correspondentes ao conjunto de teste.
3.7. Aplicando escalonamento adequado para cada modelo
Python
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

X_train_std = scaler_standard.fit_transform(X_train)
X_test_std = scaler_standard.transform(X_test)

X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

Inicializando os escalonadores:
StandardScaler(): Este escalonador realiza a padronização dos dados, ou seja, transforma os dados para que eles tenham média zero e desvio padrão igual a 1. A padronização é útil para algoritmos que dependem da distância, como KNN, SVM, etc., e quando as variáveis possuem distribuições com diferentes escalas (por exemplo, altura e peso).
MinMaxScaler(): Este escalonador realiza a normalização dos dados, ou seja, transforma os dados para que fiquem em um intervalo específico, geralmente entre 0 e 1. A normalização é útil quando você quer que todas as variáveis tenham o mesmo peso e estejam em um intervalo semelhante, sem distorcer as distribuições.
Padronização (usando StandardScaler):
fit_transform(X_train):
O método fit() calcula a média e o desvio padrão de cada feature no conjunto de dados de treino (X_train).
O método transform() aplica a padronização utilizando a média e o desvio padrão calculados no fit().
Resultado: X_train_std contém os dados de treinamento padronizados, ou seja, com média 0 e desvio padrão 1.
transform(X_test): Aqui, apenas transformamos os dados de teste (X_test) com base na média e desvio padrão calculados nos dados de treinamento (X_train). Isso é importante para garantir que os dados de teste sejam escalonados com os mesmos parâmetros dos dados de treinamento.
Resultado: X_test_std contém os dados de teste padronizados.
Normalização (usando MinMaxScaler):
fit_transform(X_train):
O método fit() calcula o valor mínimo e o valor máximo de cada feature no conjunto de dados de treino (X_train).
O método transform() aplica a normalização para que todos os valores fiquem no intervalo [0, 1].
Resultado: X_train_minmax contém os dados de treinamento normalizados.
transform(X_test): Aqui, os dados de teste (X_test) são normalizados usando os valores mínimo e máximo calculados a partir dos dados de treinamento (X_train).
Resultado: X_test_minmax contém os dados de teste normalizados.
3.8. Definir os Modelos
Python
models = {
    "Naive Bayes": (MultinomialNB(), X_train_minmax, X_test_minmax),
    "Árvore de Decisão": (DecisionTreeClassifier(random_state=42), X_train, X_test),
    "KNN": (KNeighborsClassifier(), X_train_std, X_test_std),
    "Regressão Logística": (LogisticRegression(max_iter=1000, random_state=42), X_train_std, X_test_std),
    "Floresta Aleatória": (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
    "SVM": (SVC(kernel="linear", random_state=42), X_train, X_test)
}

O dicionário models armazena os seguintes modelos de aprendizado de máquina:
Naive Bayes
Árvore de Decisão
KNN (K-Nearest Neighbors)
Regressão Logística
Floresta Aleatória
SVM
Cada chave do dicionário (por exemplo, "Naive Bayes") representa o nome do modelo, e os valores associados a cada chave são tuplas com três elementos:
O modelo em si (por exemplo, MultinomialNB() para Naive Bayes).
O conjunto de dados de treino (por exemplo X_train_minmax ou X_train_std), dependendo do modelo.
O conjunto de dados de teste (por exemplo X_test_minmax ou X_test_std), depende do modelo.
O que Significa Cada Componente:
Naive Bayes (MultinomialNB()): Usa o classificador Naive Bayes para dados categóricos/discretos e é alimentado com dados escalados usando MinMaxScaler.
Árvore de Decisão (DecisionTreeClassifier): Um modelo baseado em regras de decisão, com random_state=42 para garantir reprodutibilidade. Usa dados padronizados (StandardScaler).
KNN (KNeighborsClassifier()): Um modelo baseado em vizinhos mais próximos, também usando dados padronizados.
Regressão Logística (LogisticRegression()): Um modelo de classificação linear, ajustado para 1000 iterações e com random_state=42. Usa dados padronizados.
Floresta Aleatória (RandomForestClassifier()): Um modelo baseado em múltiplas árvores de decisão, com 100 estimadores e random_state=42. Usa os dados originais.
SVM (SVC(kernel="linear")): Um classificador baseado em Máquinas de Vetores de Suporte (SVM) com um kernel linear, também utilizando os dados originais.
Resumo do Processo:
O dicionário models contém diferentes algoritmos de aprendizado de máquina (Naive Bayes, Árvore de Decisão, KNN, Regressão Logística, Floresta Aleatória e SVM) com seus respectivos conjuntos de treino e teste. Cada modelo utiliza um tipo adequado de escalonamento:
Naive Bayes usa normalização (MinMaxScaler) para evitar valores negativos.
KNN e Regressão Logística utilizam padronização (StandardScaler) para melhorar a performance.
Árvore de Decisão, Floresta Aleatória e SVM são treinados com os dados originais, pois não são sensíveis a escala.
3.9. Treinar e Avaliar os Modelos
Python
for name, (model, X_train_data, X_test_data) in models.items():
    print("=" * 50)
    print(f"Modelo: {name}")
    print("=" * 50)

    model.fit(X_train_data, y_train)
    y_pred = model.predict(X_test_data)

   print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred), "\n")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

Estrutura do Laço for (Iterando sobre os Modelos):
 Python
for name, (model, X_train_data, X_test_data) in models.items():


O laço for percorre todos os itens do dicionário models.
O dicionário models contém vários modelos de aprendizado de máquina.
items() retorna uma sequência de pares chave-valor do dicionário, onde:
name é a chave (o nome do modelo, como "Naive Bayes").
(model, X_train_data, X_test_data) é o valor associado à chave, que é uma tupla contendo:
O modelo (como MultinomialNB(), DecisionTreeClassifier(), etc.).
O conjunto de dados de treino (X_train_data).
O conjunto de dados de teste (X_test_data).
Imprimindo Informações sobre o Modelo:
 Python
print("=" * 50)
print(f"Modelo: {name}")
print("=" * 50)


Este bloco imprime uma linha de separação com 50 sinais de igual (=), seguida pelo nome do modelo (name), que é impresso na tela para indicar qual modelo está sendo treinado e avaliado.
Isso ajuda a visualizar e separar os resultados de cada modelo, tornando a saída do código mais organizada.
Treinando o Modelo:
 Python
model.fit(X_train_data, y_train)


O método fit() é usado para treinar o modelo no conjunto de dados de treino.
X_train_data: Conjunto de dados de entrada (features) para o treinamento.
y_train: Rótulos de classe (a variável alvo) para o treinamento.
Durante o treinamento, o modelo ajusta seus parâmetros internos para aprender os padrões e fazer previsões precisas.
Fazendo Previsões:
 Python
y_pred = model.predict(X_test_data)


O método predict() é usado para fazer previsões no conjunto de dados de teste.
X_test_data: Conjunto de dados de entrada (features) para o qual o modelo precisa fazer previsões.
O modelo já foi treinado com o X_train_data e agora usa o conhecimento adquirido para prever as classes dos dados de teste.
y_pred armazena as previsões feitas pelo modelo.
Exibindo métricas organizadas:
 Python
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}\n")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred), "\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))


Esse código exibe as métricas de avaliação do modelo após fazer as previsões no conjunto de dados de teste. Ele calcula e imprime a acurácia, a matriz de confusão e o relatório de classificação, permitindo avaliar a performance do modelo de maneira detalhada.
accuracy_score(y_test, y_pred): Essa função calcula a acurácia do modelo, que é a proporção de previsões corretas entre o total de previsões feitas.
y_test: Os rótulos reais do conjunto de dados de teste.
y_pred: As previsões feitas pelo modelo para o conjunto de dados de teste.
O formato :.4f é usado para exibir a acurácia com 4 casas decimais.
Após calcular a acurácia, ele imprime o valor seguido de uma quebra de linha (\n).
confusion_matrix(y_test, y_pred): Mostra a matriz de confusão, que indica os acertos e erros do modelo (verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos).
classification_report(y_test, y_pred): Exibe um relatório detalhado com métricas de precisão, recall e F1-score para cada classe.
3.10. Seleção de Hiperparâmetros
Python
param_grid_rf = {
    'n_estimators': [10, 100],
    'max_depth': list(range(2, 31, 2))
}

param_grid_lr = {
    'C': np.logspace(-3, 3, 30)
}

Esse código define grades de hiperparâmetros para ajuste fino de modelos de Floresta Aleatória (Random Forest) e Regressão Logística (Logistic Regression).
Dicionário param_grid_rf (Random Forest):
'n_estimators': Número de árvores na floresta. Aqui, o código testa 10 e 100 árvores.
'max_depth': Profundidade máxima das árvores, variando de 2 a 30 em incrementos de 2.
Objetivo: Encontrar a melhor combinação de número de árvores (n_estimators) e profundidade máxima (max_depth) para otimizar o desempenho do modelo.
Dicionário param_grid_lr (Regressão Logística):
'C': Parâmetro de regularização da Regressão Logística.
np.logspace(-3, 3, 30): Gera 30 valores espaçados logaritmicamente entre 10⁻³ (0.001) e 10³ (1000).
Objetivo: Ajustar a regularização da regressão logística para encontrar o melhor valor de C.
Resumo do Processo:
Essas grades de hiperparâmetros normalmente são passadas para métodos como GridSearchCV ou RandomizedSearchCV, que testam diferentes combinações e selecionam a melhor com base em métricas como acurácia ou F1-score.
3.11. Ajuste dos Modelos
Python
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)

Ajuste do modelo Random Forest:
RandomForestClassifier(random_state=42): Modelo de Floresta Aleatória.
param_grid_rf: O dicionário de hiperparâmetros definido anteriormente.
cv=5: Validação cruzada com 5 folds (divide os dados em 5 partes e treina o modelo 5 vezes para avaliar o desempenho médio).
scoring='accuracy': A métrica usada para avaliar o modelo é acurácia.
n_jobs=-1: Utiliza todos os núcleos disponíveis do processador para acelerar o processamento.
Ajuste do modelo de Regressão Logística:
LogisticRegression(random_state=42, max_iter=1000):
random_state=42: Garante reprodutibilidade dos experimentos.
max_iter=1000: Define um número maior de iterações para garantir convergência.
param_grid_lr: O dicionário de hiperparâmetros para C (regularização).
cv=5: Validação cruzada com 5 folds.
scoring='accuracy': Usa acurácia como métrica.
n_jobs=-1: Usa todos os núcleos disponíveis.
Resumo do Processo:
Testa todas as combinações de hiperparâmetros da grade (param_grid_rf e param_grid_lr).
Treina e avalia cada configuração usando validação cruzada.
Seleciona os melhores hiperparâmetros que maximizam a acurácia.
3.12. Treinando os modelos com os melhores hiperparâmetros
Python
grid_rf.fit(X_train, y_train)
grid_lr.fit(X_train_std, y_train)

best_rf = grid_rf.best_estimator_
best_lr = grid_lr.best_estimator_

print("Melhores hiperparâmetros para RandomForest:", grid_rf.best_params_)
print("Melhores hiperparâmetros para Regressão Logística:", grid_lr.best_params_)

Agora o código treina os modelos e seleciona os melhores hiperparâmetros encontrados pelo GridSearchCV.
Treinando o modelo Random Forest:
grid_rf.fit(X_train, y_train):
Realiza a busca pelos melhores hiperparâmetros com base na acurácia.
Treina o modelo de Random Forest nas combinações de param_grid_rf.
Usa os dados não padronizados (X_train e y_train), pois a Floresta Aleatória não é sensível à escala dos dados.
Treinando o modelo de Regressão Logística:
grid_lr.fit(X_train_std, y_train):
Treina a Regressão Logística com validação cruzada.
Usa dados padronizados (X_train_std).
Isso é necessário porque a Regressão Logística é sensível à escala dos dados.
A padronização geralmente é feita com StandardScaler(), que transforma os dados para média 0 e desvio padrão 1.
Obtendo os melhores modelos:
best_rf = grid_rf.best_estimator_: Retorna a melhor configuração encontrada para o modelo de Random Forest.
best_lr = grid_lr.best_estimator_: Retorna a melhor configuração para o modelo de Regressão Logística.
Imprimindo os melhores hiperparâmetros:
print("Melhores hiperparâmetros para RandomForest:", grid_rf.best_params_)
print("Melhores hiperparâmetros para Regressão Logística:", grid_lr.best_params_)
Aqui só imprimimos os melhores hiperparâmetros.
3.13. Avaliação dos modelos com os melhores hiperparâmetros
Python
models = {
    "Floresta Aleatória": best_rf,
    "Regressão Logística": best_lr
}

results = {}

for name, model in models.items():
    print("=" * 50)
    print(f"Modelo: {name}")
    print("=" * 50)

    if name == "Regressão Logística":
        y_pred = model.predict(X_test_std)
    else:
        y_pred = model.predict(X_test)

    results[name] = y_pred

    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred), "\n")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

Criação de um dicionário com os modelos:
 Python
models = {
    "Floresta Aleatória": best_rf,
    "Regressão Logística": best_lr
}


Armazena os melhores modelos encontrados pelo GridSearchCV para avaliação.
Dicionário para armazenar previsões:
 Python
results = {}


Usado para salvar as previsões (y_pred) de cada modelo, que podem ser úteis para testes de hipótese depois.
Loop para avaliação dos modelos:
 Python
for name, model in models.items():


Itera sobre cada modelo e faz a avaliação.
Exibição do nome do modelo:
 Python
print("=" * 50)
print(f"Modelo: {name}")
print("=" * 50)


Apenas para formatação e melhor visualização dos resultados.
Previsões com os modelos:
 Python
if name == "Regressão Logística":
    y_pred = model.predict(X_test_std)
else:
    y_pred = model.predict(X_test)


Regressão Logística exige dados padronizados (X_test_std).
Floresta Aleatória pode trabalhar diretamente com os dados brutos (X_test).
Armazena os resultados das previsões:
 Python
results[name] = y_pred


Guarda os resultados de cada modelo para análise posterior.
Métricas de avaliação:
 Python
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}\n")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred), "\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))


accuracy_score(y_test, y_pred): Calcula a acurácia do modelo.
confusion_matrix(y_test, y_pred): Mostra a matriz de confusão, que indica os acertos e erros do modelo.
classification_report(y_test, y_pred): Exibe precisão, recall e F1-score para cada classe.
Agora, este código avalia o desempenho dos modelos treinados nos dados de teste e exibe métricas importantes como acurácia, matriz de confusão e relatório de classificação.
3.14. Visualização da Importância das Características (Floresta Aleatória)
Python
feature_importances = best_rf.feature_importances_
feature_names = X.columns

sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[sorted_indices[:10]], y=np.array(feature_names)[sorted_indices[:10]], palette="viridis")
plt.xlabel("Importância da Característica")
plt.ylabel("Características")
plt.title("Top 10 Características Mais Importantes (Floresta Aleatória)")
plt.show()

O código a seguir visualiza a importância das características no modelo de Floresta Aleatória (best_rf). Ele gera um gráfico de barras mostrando as 10 características mais relevantes para as previsões do modelo.
Obtendo a importância das características:
feature_importances = best_rf.feature_importances_: Retorna um array com a importância de cada variável no modelo.
feature_names = X.columns: Obtém os nomes das colunas para referenciar cada característica.
Ordenando as características por importância:
sorted_indices = np.argsort(feature_importances)[::-1]:
np.argsort(feature_importances): Retorna os índices das características ordenadas por importância (do menor para o maior).
[::-1]: Inverte a ordem para ficar do maior para o maior.
Criando o gráfico de barras:
plt.figure(figsize=(10, 6)): Define o tamanho do gráfico.
sns.barplot(...): Cria um gráfico de barras com:
Eixo X: Importância das características.
Eixo Y: Nome das características.
palette="viridis": Define uma paleta de cores.
Personalizando o gráfico:
plt.xlabel("Importância da Característica")
plt.ylabel("Características")
plt.title("Top 10 Características Mais Importantes (Floresta Aleatória)")
plt.show()
Adiciona título e rótulos para melhorar a interpretação.
3.15. Análise da Distribuição das Características por Classe
Python
df["Prediction"] = df["Prediction"].astype(str)
valid_features = [feature for feature in feature_names if df[feature].var() > 0]
selected_features = valid_features[:5]

for feature in selected_features:
    plt.figure(figsize=(8, 4))
   sns.kdeplot(data=df, x=feature, hue="Prediction", common_norm=False, fill=True, alpha=0.3, warn_singular=False)
    plt.title(f"Distribuição de {feature} por Classe")
    plt.xlabel(feature)
    plt.ylabel("Densidade")
    plt.legend(title="Classe", labels=["Não Spam (0)", "Spam (1)"])
    plt.show()
O código a seguir garante a categorização da variável de previsão, filtra variáveis com variação e gera gráficos de distribuição para visualizar como algumas características se comportam em relação às classes.
Garantir que a coluna ‘Prediction’ seja categórica:
 Python
df["Prediction"] = df["Prediction"].astype(str)


Converte a coluna Prediction para string, garantindo que seja tratada como uma categoria discreta ao gerar gráficos.
Filtrar características relevantes:
 Python
valid_features = [feature for feature in feature_names if df[feature].var() > 0]


Objetivo: Remover variáveis constantes (sem variação) que podem ser irrelevantes para a análise.
df[feature].var() > 0: Mantém apenas colunas cuja variância seja positiva (ou seja, cujos valores realmente variam).
Selecionar as 5 características mais importantes:
 Python
selected_features = valid_features[:5]


Escolhe as 5 primeiras características da lista filtrada para visualizar sua distribuição.
Gerar gráficos de distribuição das variáveis por classe:
 Python
for feature in selected_features:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df, x=feature, hue="Prediction", common_norm=False, fill=True, alpha=0.3, warn_singular=False)
    plt.title(f"Distribuição de {feature} por Classe")
    plt.xlabel(feature)
    plt.ylabel("Densidade")
    plt.legend(title="Classe", labels=["Não Spam (0)", "Spam (1)"])
    plt.show()


for feature in selected_features: Gera um gráfico para cada variável selecionada.
sns.kdeplot(...):
x=feature: A variável sendo analisada.
hue="Prediction": As curvas são separadas por classe (Spam ou Não Spam).
common_norm=False: Mantém densidades separadas para cada classe.
fill=True, alpha=0.3: Preenche as curvas para melhor visualização.
Legendas e títulos:
O título do gráfico inclui o nome da variável analisada.
A legenda diferencia as classes (0 = Não Spam, 1 = Spam).
3.16. Visualização da Importância das Características (Regressão Logística)
Python
coefficients = best_lr.coef_[0]
importances = np.abs(coefficients)
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_indices[:10]], y=np.array(feature_names)[sorted_indices[:10]], palette="viridis")
plt.xlabel("Importância da Característica (Valor Absoluto do Coeficiente)")
plt.ylabel("Características")
plt.title("Top 10 Características Mais Importantes (Regressão Logística)")
plt.show()

O código a seguir gera um gráfico de barras com as 10 características mais importantes no modelo de Regressão Logística, baseado no valor absoluto dos coeficientes do modelo.
Obtendo os coeficientes do modelo:
 Python
coefficients = best_lr.coef_[0]


best_lr.coef_: Retorna os coeficientes do modelo de Regressão Logística.
[0]: Como a regressão logística binária tem apenas uma saída, pegamos os coeficientes correspondentes à única classe do modelo.
Calculando a importância das características:
 Python
importances = np.abs(coefficients)


Como os coeficientes podem ser positivos ou negativos, usamos np.abs(coefficients), pois a magnitude indica a influência da variável, independentemente do seu sinal.
Ordenando as características por importância:
 Python
sorted_indices = np.argsort(importances)[::-1]


np.argsort(importances): Retorna os índices das características ordenadas do menor para o maior.
[::-1]: Inverte a ordem para exibir do maior para o menor.
Gerando o gráfico de barras:
 Python
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_indices[:10]], y=np.array(feature_names)[sorted_indices[:10]], palette="viridis")


sns.barplot(...):
Eixo X: Importância das características (coeficientes absolutos).
Eixo Y: Nome das características.
palette="viridis": Define uma paleta de cores para melhorar a visualização.
Personalizando o gráfico:
 Python
plt.xlabel("Importância da Característica (Valor Absoluto do Coeficiente)")
plt.ylabel("Características")
plt.title("Top 10 Características Mais Importantes (Regressão Logística)")
plt.show()


Adiciona rótulos e título para facilitar a interpretação.
3.17. Teste t Pareado para Comparação de Modelos
Python
scores_lr = cross_val_score(best_lr, X, y, cv=5, scoring='accuracy')
scores_rf = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
df_scores = pd.DataFrame({'Regressão Logística': scores_lr, 'Floresta Aleatória': scores_rf})
t_statistic, p_value = stats.ttest_rel(df_scores['Regressão Logística'], df_scores['Floresta Aleatória'], alternative='less')

alpha = 0.05
if p_value < alpha:
    print("Rejeitamos a hipótese nula: a Regressão Logística tem desempenho significativamente menor que a Floresta Aleatória.")
else:
    print("Não rejeitamos a hipótese nula: não há evidências suficientes para concluir que a Regressão Logística tem desempenho menor que a Floresta Aleatória.")

O código a seguir compara estatisticamente os desempenhos da Regressão Logística e da Floresta Aleatória usando um teste t pareado com validação cruzada.
Realizando a Validação Cruzada:
 Python
scores_lr = cross_val_score(best_lr, X, y, cv=5, scoring='accuracy')
scores_rf = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')


cross_val_score(modelo, X, y, cv=5, scoring='accuracy'):
Avalia cada modelo usando validação cruzada k-fold (k=5).
Calcula a acurácia em cada uma das 5 subdivisões dos dados.
Retorna um array com 5 valores de acurácia, um para cada fold.
Criando um DataFrame para os resultados:
 Python
df_scores = pd.DataFrame({'Regressão Logística': scores_lr, 'Floresta Aleatória': scores_rf})


Organiza os valores de acurácia em um DataFrame para facilitar a análise.
Executando o Teste t Pareado:
 Python
t_statistic, p_value = stats.ttest_rel(df_scores['Regressão Logística'], df_scores['Floresta Aleatória'], alternative='less')


stats.ttest_rel():
Testa se há diferença estatística significativa entre os desempenhos dos modelos.
Como cada fold testa os dois modelos nos mesmos dados, o teste t é pareado.
alternative='less':
Testa se a Regressão Logística tem um desempenho significativamente menor que a Floresta Aleatória.
Hipótese nula (H0): Ambos os modelos têm desempenhos semelhantes.
Hipótese alternativa (Ha): A Floresta Aleatória é significativamente melhor
Interpretando o Resultado:

 Python
alpha = 0.05  # Nível de significância
if p_value < alpha:
    print("Rejeitamos a hipótese nula: a Regressão Logística tem desempenho significativamente menor que a Floresta Aleatória.")
else:
    print("Não rejeitamos a hipótese nula: não há evidências suficientes para concluir que a Regressão Logística tem desempenho menor que a Floresta Aleatória.")


Se p_value < 0.05: Há evidências estatísticas de que a Regressão Logística tem desempenho inferior.
Se p_value >= 0.05: Não há evidências suficientes para afirmar que a Floresta Aleatória seja melhor.
3.18. Comparação de Acurácias e Matriz de Confusão (Gráfico)
Python
modelos = list(models.keys())
acuracias = [
    accuracy_score(y_test, best_rf.predict(X_test)),
    accuracy_score(y_test, best_lr.predict(X_test_std))
]

fig, ax = plt.subplots(figsize=(10, 6))
barras = ax.bar(modelos, acuracias, color=['skyblue', 'lightcoral'])

for barra, acuracia in zip(barras, acuracias):
    ax.text(barra.get_x() + barra.get_width() / 2, acuracia + 0.01, f'{acuracia:.4f}', ha='center', va='bottom')

matrizes_confusao = [
    confusion_matrix(y_test, results['Floresta Aleatória']),
    confusion_matrix(y_test, results['Regressão Logística'])
]

for i, modelo in enumerate(modelos):
    matriz = matrizes_confusao[i]
    texto_matriz = f'Matriz de Confusão:\n{np.array(matriz)}'
    ax.text(i, 0.1, texto_matriz, ha='center', va='bottom', fontsize=8)

ax.set_ylim(0, 1.1)
ax.set_title('Comparação da Acurácia dos Modelos')
ax.set_xlabel('Modelos')
ax.set_ylabel('Acurácia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

O código a seguir calcula as acurácias dos modelos no conjunto de teste.
 Python
acuracias = [
    accuracy_score(y_test, best_rf.predict(X_test)),  # Floresta Aleatória
    accuracy_score(y_test, best_lr.predict(X_test_std))  # Regressão Logística
]


best_rf.predict(X_test): Faz previsões no conjunto de teste não padronizado usando a Floresta Aleatória.
best_lr.predict(X_test_std): Faz previsões no conjunto de teste padronizado para a Regressão Logística (já que esse modelo é sensível à escala).
accuracy_score(y_test, previsões): Compara as previsões com os valores reais (y_test) e calcula a acurácia.
O código a seguir cria uma lista contendo os nomes dos modelos que você está avaliando.
 Python
modelos = list(models.keys())


models: É um dicionário contendo os modelos treinados, definido anteriormente.
models.keys(): Retorna as chaves do dicionário, ou seja, os nomes dos modelos.
list(...): Converte essas chaves em uma lista.
O código a seguir gera um gráfico de barras comparando a acurácia dos modelos Floresta Aleatória e Regressão Logística, além de incluir a matriz de confusão como anotação no gráfico.
Criando o gráfico de barras:
 Python
fig, ax = plt.subplots(figsize=(10, 6))
barras = ax.bar(modelos, acuracias, color=['skyblue', 'lightcoral'])


plt.subplots(figsize=(10, 6)): Cria uma figura e um eixo para o gráfico com um tamanho adequado.
ax.bar(modelos, acuracias, color=['skyblue', 'lightcoral']):
Cria barras para cada modelo com suas respectivas acurácias.
Atribui cores diferentes para cada barra.
Adicionando os valores de acurácia no topo das barras:
 Python
for barra, acuracia in zip(barras, acuracias):
    ax.text(barra.get_x() + barra.get_width() / 2, acuracia + 0.01, f'{acuracia:.4f}', ha='center', va='bottom')


ax.text(...) escreve o valor da acurácia acima da barra correspondente.
Calculando e adicionando a Matriz de Confusão:
 Python
matrizes_confusao = [
    confusion_matrix(y_test, results['Floresta Aleatória']),  # Floresta Aleatória
    confusion_matrix(y_test, results['Regressão Logística'])  # Regressão Logística
]

for i, modelo in enumerate(modelos):
    matriz = matrizes_confusao[i]
    texto_matriz = f'Matriz de Confusão:\n{np.array(matriz)}'
    ax.text(i, 0.1, texto_matriz, ha='center', va='bottom', fontsize=8)


confusion_matrix(y_test, results['Floresta Aleatória']): Calcula a matriz de confusão para cada modelo.
ax.text(...) adiciona a matriz de confusão abaixo das barras no gráfico, facilitando a interpretação dos erros e acertos.
Ajustando a aparência do gráfico:
 Python
ax.set_ylim(0, 1.1)
ax.set_title('Comparação da Acurácia dos Modelos')
ax.set_xlabel('Modelos')
ax.set_ylabel('Acurácia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


Define o título do gráfico, os rótulos dos eixos e melhora a legibilidade dos nomes dos modelos.
4. Conclusão
Este projeto demonstrou a aplicação de diversas técnicas de aprendizado de máquina para a classificação de e-mails como spam ou não spam. A otimização de hiperparâmetros e a avaliação detalhada dos modelos permitiram identificar os algoritmos mais eficazes para essa tarefa. A análise da importância das características e a visualização da distribuição dos dados forneceram insights valiosos sobre os padrões presentes nos e-mails.
5. Próximos Passos
Explorar técnicas de engenharia de características para criar novas variáveis que possam melhorar o desempenho dos modelos.
Investigar outros algoritmos de classificação e técnicas de aprendizado profundo para aprimorar ainda mais a precisão.
Implementar o modelo em um ambiente de produção para classificar e-mails em tempo real.
Realizar testes adicionais para avaliar a robustez e a generalização do modelo em diferentes conjuntos de dados.
