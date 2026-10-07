import pandas as pd
import numpy as np
import joblib

#Módulos de pré-processamento e divisão
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Algoritmos de machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Métricas de avaliação
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Biblioteca para balancemento de dados
from imblearn.over_sampling import SMOTE

#Passo 1 carregamento da base bruta
print("Carregando a base de dados...")
caminho_csv = "esus-vepi.LeitoOcupacao_2022.csv"
df = pd.read_csv(caminho_csv)

#Passo 2 limpeza de dados
#Colunas irrelevantes ou quase 100% nulas identificadas no EDA (Análise exploratória)
Colunas_para_remover = [
    'Unnamed: 0', '_id', '_created_at', '_updated_at', '_p_usuario',
    'ocupacaoSuspeitoCli', 'ocupacaoSuspeitoUti',
    'ocupacaoConfirmadoCli', 'ocupacaoConfirmadoUti'
]
# errors='ignore' evita que o código quebre caso alguma dessas colunas já não exista
df_limpo = df.drop(columns=Colunas_para_remover, errors='ignore')

#Filtro de qualidade: removemos linhas onde todas as métricas de ocupação são 0 (registros inválidos)
colunas_metricas = [
    'ocupacaoCovidUti', 'ocupacaoCovidCli',
    'ocupacaoHospitalarUti', 'ocupacaoHospitalarCli'
]
df_limpo = df_limpo[df_limpo[colunas_metricas].sum(axis=1) > 0]

#Preenchemos eventuais células vazias (NaN) restantes com 0
df_limpo = df_limpo.fillna(0)

#Passo 3 engenharia de atributos
#criamos variáveis calculados que ajudam os modelos a entender a proporção e pressão no hospital
df_feat = df_limpo.copy()

#atributo 1 soma total de leitos ocupados no hospital naquele registro
df_feat['totalOcupacao'] = (
    df_feat['ocupacaoCovidUti'] +
    df_feat['ocupacaoCovidCli'] +
    df_feat['ocupacaoHospitalarUti'] +
    df_feat['ocupacaoHospitalarCli']
)

#atributo 2 proporção de leitos de UTI COVID em relação ao total ocupado (evita divisão por zero somando 1e-5)
df_feat['proporcaoCovidUti'] = df_feat['ocupacaoCovidUti'] / (df_feat['totalOcupacao'] + 1e-5)

#atributo 3 razão entre altas concedidas e ocupação total
df_feat['razaoAltasOcupacao'] = df_feat['saidaConfirmadaAltas'] / (df_feat['totalOcupacao'] + 1e-5)

#seleção das variáveis finais de entrada (x) e do nosso alvo/target (y)
x = df_feat[[
    'ocupacaoCovidUti',
    'ocupacaoCovidCli',
    'ocupacaoHospitalarUti',
    'ocupacaoHospitalarCli',
    'saidaConfirmadaAltas',
    'totalOcupacao',
    'proporcaoCovidUti',
    'razaoAltasOcupacao'
]]

#O alvo y será 1 se houver ao menos 1 óbito confirmados, e 0 se não houver óbitos
y = (df_feat['saidaConfirmadaObitos'] > 0).astype(int)

#Passo 4 divisão em treino e teste
#stratify=y garante que proporção original entre óbitos (1) e altas (0)
#seja mantida idêntica tanto no treino (80%) quanto no teste (20%)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

#Passo 5 aplicação do smote e padronização
print("Aplicando SMOTE para balancear a classe de óbitos no conjunto de treino...")
smote = SMOTE(random_state=42)
#reamostramos apenas o conjunto de treino
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

#O StandardScaler ajusta a escala dos dados (média 0, desvio padrão 1)
#E essencial para a regresão logistica, pois ela calcula a distância matemáticas entre os atributos
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_res) #aprende a escala do treino e transforma
x_test_scaled = scaler.transform(x_test) #aplica a mesma escala ao teste

#Passo 6 comparação e avaliação de modelos
print("Treinando e comparando os algoritmos...")

modelos = {
    "Árvore de decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Regressão Logistica": LogisticRegression(max_iter=1000, random_state=42)
}

resultados = []

for nome, modelo in modelos.items():
    #a regressão logistica precisa dos dados escalados (StandarScaler)
    if nome == "Regressão Logistica":
        modelo.fit(x_train_scaled, y_train_res)
        y_pred = modelo.predict(x_test_scaled)
    else:
        #Árvore de decissão e Random Forest não dependem de escala
        modelo.fit(x_train_res, y_train_res)
        y_pred = modelo.predict(x_test)

    #calculo das metricas comparativas
    resultados.append({
        "Modelo": nome,
        "Acurácia": f"{accuracy_score(y_test, y_pred) * 100:.2f}%", # % geral de acertos
        "Precissão": f"{precision_score(y_test, y_pred, zero_division=0) * 100:.2f}%", #quando preve obito, quantos acerta?
        "Recall": f"{recall_score(y_test, y_pred, zero_division=0) * 100:.2f}%", #de todos os obitos reais, quantos ele pegou?
        "F1-Score": f"{f1_score(y_test, y_pred, zero_division=0) * 100:.2f}%" #media harmonica entre precissão e recall
    })

#Passo 7 exibição e exportação
df_resultados = pd.DataFrame(resultados)
print("\n=== TABELA COMPRATIVA DE DESEMPENHO ===")
print(df_resultados.to_string(index=False))

#Salvamos o Random Forest getalmente o modelo mais robusto para tabelas
modelo_escolhido = modelos["Random Forest"]
joblib.dump(modelo_escolhido, 'Modelo_Covid.pkl')
print("\n[SUCESSO] Modelo 'Modelo_Covid.pkl' atualizado com sucesso")