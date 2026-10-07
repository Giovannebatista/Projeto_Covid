from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional

# Importação do SQLAlchemy banco de dados
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Configuração do banco de dados SQLite + SQLAlchemy
DATABASE_URL = "sqlite:///./historico.db"

# connect_args={"check_same_thead": False} e necessário para o SQLite no FastAPI
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelo da tabela no banco de dados
class historicoPrevissaoModel(Base):
    __tablename__ = "historico_previsoes"

    id = Column(Integer, primary_key=True, index=True)
    data_hora = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ocupacaoCovidUti = Column(Float)
    ocupacaoCovidCli = Column(Float)
    ocupacaoHospitalarUti = Column(Float)
    ocupacaoHospitalarCli = Column(Float)
    saidaConfirmadaAltas = Column(Float)
    totalOcupacao = Column(Float)
    predicao = Column(Integer) # 0 = sem obito/ baixo risco, 1 = risco/obito
    probabilidade_obito = Column(Float) # probabilidade calculada pelo modelo ml

# Instancia e cria a tebela no banco 'historico.db se ela ainda não existir
Base.metadata.create_all(bind=engine)

def get_db():
    """
    Injeção de Dependência para o FastAPI.
    Abre uma sessão com o banco de dados a cada requisição e garante 
    o fechamento correto da conexão ao final da execução (mesmo que ocorra erro).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Carregamento do modelo de machine learning

# bloco try/except para evitar que o servidor caia se o arquivo .pkl estiver ausente
try:
    modelo = joblib.load('Modelo_Covid.pkl')
    print("[INFO] Modelo de machine learning 'Modelo_covid.pkl' carregando com sucesso")
except Exception as e:
    modelo = None
    print(f"[Alerta critico] não foi possivel carregar o modelo ml: {e}")


# schemas de validação de dados
class PrevissaoEntrada(BaseModel):
    """
    Garante que os dados enviados via formulário/JSON no POST /predict estejam corretos.
    O uso do 'Field(ge=0)' impede a entrada de números negativos (Validação de Dados).
    """
    ocupacaoCovidUti: float = Field(
        ..., ge=0, description="Quantidade de leitos de UTI COVID ocupados. Deve ser >= 0"
    )
    ocupacaoCovidCli: float = Field(
        ..., ge=0, description="Quantidade de leitos de Clinica COVID ocupados. Deve ser >= 0"
    )
    ocupacaoHospitalarUti: float = Field(
        ..., ge=0, description="Quantidade de leitos de UTI Geral ocupados. Deve ser >= 0"
    )
    ocupacaoHospitalarCli: float = Field(
        ..., ge=0, description="Quantidade de leitos de Clinica Geral ocupados. Deve ser >= 0"
    )
    saidaConfirmadaAltas: float = Field(
        ..., ge=0, description="Número acumulado de altas hospitalares. Deve ser >= 0"
    )

    # configuração de exemplo visual para preencher a documentação interativa /docs
    class Config:
        json_schema_extra = {
            "example": {
                "ocupacaoCovidUti": 15.0,
                "ocupacaoCovidCli": 25.0,
                "ocupacaoHospitalarUti": 50.0,
                "ocupacaoHospitalarCli": 120.0,
                "saidaConfirmadaAltas": 85.0
            }
        }

class PrevissaoResposta(BaseModel):
    """
    Define a estrutura formal da resposta JSON enviada ao cliente após uma simulação.
    """
    id_simulacao: int
    data_hora: datetime
    risco_obito_detectado: int
    probabilidade_obito_pct: float
    mensagem: str

class historicoItem(BaseModel):
    """
    Define como cada item do histórico deve ser retornado no endpoint GET /historico.
    """
    id: int
    data_hora: datetime
    ocupacaoCovidUti: float
    ocupacaoCovidUti: float
    ocupacaoHospitalarUti: float
    ocupacaoHospitalarCli: float
    saidaConfirmadaAltas: float
    totalOcupacao: float
    predicao: int
    probabilidade_obito: float

    # from_attributes = True permite converter objetos do SQLlchemy direto para JSON
    class Config:
        from_attributes = True

# Inicialização e documentação da API
app = FastAPI(
    title="API de Predição e Gestão Hospitalar - COVID-19",
    description="""
    ## Sistema Inteligente de Alerta e Previsão Hospitalar

    Esta API conecta um modelo de **Machine Learning (Random Forest)** treinado com dados 
    do ESUS-VEPI a um banco de dados para previsão em tempo real do risco de óbitos.

    ### Recursos Principais:
    * **Validação Rígida:** Proteção contra dados inconsistentes/negativos.
    * **Engenharia de Atributos Interna:** Calcula taxas de ocupação relativas dinamicamente.
    * **Auditoria de Simulações:** Armazena cada predição no banco **SQLite**.
    """,
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modulo rotas e endpoints da API

@app.get("/", tags=["Status"])
def root():
    """
    Endpoint simples de verificação do estado da aplicação (Health Check).
    """
    return {
        "status": "online",
        "mensagem": "API de Predicação COVID-19 operando normalmente",
        "documentcao_swagger": "/docs"
    }

@app.post(
    "/predict",
    response_model=PrevissaoResposta,
    status_code=status.HTTP_201_CREATED,
    tags=["Predição de Machine Learning"],
    summary="Executa a predição de risco e grava a simulação no Banco de Dados",
    description="""
    Recebe os dados de ocupação, calcula os indicadores derivados, submete ao algoritmo 
    e persiste a simulação no histórico.
    """
)
def realizar_previsao(dados: PrevissaoEntrada, db: Session = Depends(get_db)):
    # verificação se o modelo esta carregando no servidor
    if modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O modelo de Machine Learning não está diponivel no servidor"
        )

    # Engenharia de atributos (recria exatamente as variaveis usadas no treino do SMOTE)
    totalOcupacao = (
        dados.ocupacaoCovidUti +
        dados.ocupacaoCovidCli +
        dados.ocupacaoHospitalarUti +
        dados.ocupacaoHospitalarCli
    )
    # Soma +1e-5 (0,00001) para evitar o error fatal de divisão por zero caso totalOcupacao seja 0
    proporcaoCovidUti = dados.ocupacaoCovidUti / (totalOcupacao + 1e-5)
    razaoAltasOcupacao = dados.saidaConfirmadaAltas / (totalOcupacao + 1e-5)

    # Criação do dataframe (respeitando rigorosamente a ordem de colunas do treino)
    features = pd.DataFrame([{
        'ocupacaoCovidUti': dados.ocupacaoCovidUti,
        'ocupacaoCovidCli': dados.ocupacaoCovidCli,
        'ocupacaoHospitalarUti': dados.ocupacaoHospitalarUti,
        'ocupacaoHospitalarCli': dados.ocupacaoHospitalarCli,
        'saidaConfirmadaAltas': dados.saidaConfirmadaAltas,
        'totalOcupacao': totalOcupacao,
        'proporcaoCovidUti': proporcaoCovidUti,
        'razaoAltasOcupacao': razaoAltasOcupacao
    }])

    #linha temporaria
    print("Colunas que o modelo espera:", modelo.feature_names_in_)

    # Execução da predicação e probabilidade
    predicao = int(modelo.predict(features)[0])

    # Extrair a probabilidade de pertencer a classe 1 (Obito/Risco)
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(features)[0]
        prob_obito = float(probabilidades[1] * 100)
    else:
        prob_obito = 100.0 if predicao == 1 else 0.0

    # Gravação no banco de dados
    novo_registro = historicoPrevissaoModel(
        ocupacaoCovidUti=dados.ocupacaoCovidUti,
        ocupacaoCovidCli=dados.ocupacaoCovidCli,
        ocupacaoHospitalarUti=dados.ocupacaoHospitalarUti,
        ocupacaoHospitalarCli=dados.ocupacaoHospitalarCli,
        saidaConfirmadaAltas=dados.saidaConfirmadaAltas,
        totalOcupacao=totalOcupacao,
        predicao=predicao,
        probabilidade_obito=round(prob_obito, 2)
    )
    db.add(novo_registro) # adiciona a transação
    db.commit() # confirma e salva no banco de dados
    db.refresh(novo_registro) # atuliza o objeto com id gerado pelo banco

    # Definição da mensagem retornada
    mensagem_status = (
        "ALERTA CRÍTICO: Risco elevado de óbito detectado. Recomenda-se acompanhamento intensivo."
        if predicao == 1 else
        "SITUAÇÃO ESTÁVEL: Parâmetros hospitalares dentro dos limites de baixo risco."
    )

    # 7. Retorno estruturado conforme o 'PrevissaoResposta'
    return PrevissaoResposta(
        id_simulacao=novo_registro.id,
        data_hora=novo_registro.data_hora,
        risco_obito_detectado=predicao,
        probabilidade_obito_pct=round(prob_obito, 2),
        mensagem=mensagem_status
    )

@app.get(
    "/historico",
    response_model=List[historicoItem],
    tags=["Histórico & Auditoria"],
    summary="Consulta o histórico das últimas simulações salvas no banco",
    description="Retorna uma lista contendo as simulações já efetuadas, ordenadas da mais recente para a mais antiga."
)
def listar_historico(limit: int = 50, db: Session = Depends(get_db)):
    """
    Busca os registros no SQLite usando ORM ordenados pelo ID decrescente.
    """
    registros = (
        db.query(historicoPrevissaoModel)
        .order_by(historicoPrevissaoModel.id.desc())
        .limit(limit)
        .all()
    )
    return registros