import os
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from datetime import datetime, timezone
from typing import List

from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ---------------------------------------------------------------
# Banco de dados SQLite + SQLAlchemy
# ---------------------------------------------------------------
# Em Docker, aponte para um volume (ex.: sqlite:////data/historico.db)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./historico.db")

# check_same_thread=False é necessário para o SQLite funcionar com o FastAPI
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class HistoricoPrevisaoModel(Base):
    __tablename__ = "historico_previsoes"

    id = Column(Integer, primary_key=True, index=True)
    data_hora = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ocupacaoCovidUti = Column(Float)
    ocupacaoCovidCli = Column(Float)
    ocupacaoHospitalarUti = Column(Float)
    ocupacaoHospitalarCli = Column(Float)
    saidaConfirmadaAltas = Column(Float)
    totalOcupacao = Column(Float)
    predicao = Column(Integer)  # 0 = sem óbito / baixo risco, 1 = risco / óbito
    probabilidade_obito = Column(Float)  # em %


Base.metadata.create_all(bind=engine)


def get_db():
    """Abre uma sessão por requisição e garante o fechamento ao final."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------
# Carregamento do modelo de ML
# ---------------------------------------------------------------
try:
    modelo = joblib.load("Modelo_Covid.pkl")
    print("[INFO] Modelo 'Modelo_Covid.pkl' carregado com sucesso")
except Exception as e:
    modelo = None
    print(f"[ALERTA CRÍTICO] Não foi possível carregar o modelo: {e}")


# ---------------------------------------------------------------
# Schemas (validação de entrada e saída)
# ---------------------------------------------------------------
class PrevisaoEntrada(BaseModel):
    ocupacaoCovidUti: float = Field(..., ge=0, description="Leitos de UTI COVID ocupados (>= 0)")
    ocupacaoCovidCli: float = Field(..., ge=0, description="Leitos clínicos COVID ocupados (>= 0)")
    ocupacaoHospitalarUti: float = Field(..., ge=0, description="Leitos de UTI geral ocupados (>= 0)")
    ocupacaoHospitalarCli: float = Field(..., ge=0, description="Leitos clínicos gerais ocupados (>= 0)")
    saidaConfirmadaAltas: float = Field(..., ge=0, description="Altas hospitalares confirmadas (>= 0)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ocupacaoCovidUti": 15.0,
                "ocupacaoCovidCli": 25.0,
                "ocupacaoHospitalarUti": 50.0,
                "ocupacaoHospitalarCli": 120.0,
                "saidaConfirmadaAltas": 85.0,
            }
        }
    }


class PrevisaoResposta(BaseModel):
    id_simulacao: int
    data_hora: datetime
    risco_obito_detectado: int
    probabilidade_obito_pct: float
    total_ocupacao: float            # NOVO: o front não precisa recalcular
    proporcao_covid_uti_pct: float   # NOVO
    mensagem: str


class HistoricoItem(BaseModel):
    id: int
    data_hora: datetime
    ocupacaoCovidUti: float
    ocupacaoCovidCli: float          # CORRIGIDO: antes estava duplicado ocupacaoCovidUti
    ocupacaoHospitalarUti: float
    ocupacaoHospitalarCli: float
    saidaConfirmadaAltas: float
    totalOcupacao: float
    predicao: int
    probabilidade_obito: float

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------
app = FastAPI(
    title="API de Predição e Gestão Hospitalar - COVID-19",
    description="""
## Sistema Inteligente de Alerta e Previsão Hospitalar

API que conecta um modelo **Random Forest** treinado com dados do e-SUS VEPI
a um banco de dados para estimar o risco de óbitos em registros hospitalares.

* **Validação rígida:** bloqueia dados inconsistentes/negativos.
* **Engenharia de atributos interna:** calcula os indicadores derivados.
* **Auditoria:** grava cada predição no SQLite.
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, liste os domínios do front
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Status"])
def root():
    """Health check."""
    return {
        "status": "online",
        "modelo_carregado": modelo is not None,
        "mensagem": "API de Predição COVID-19 operando normalmente",
        "documentacao_swagger": "/docs",
    }


@app.post(
    "/predict",
    response_model=PrevisaoResposta,
    status_code=status.HTTP_201_CREATED,
    tags=["Predição de Machine Learning"],
    summary="Executa a predição de risco e grava a simulação no banco",
)
def realizar_previsao(dados: PrevisaoEntrada, db: Session = Depends(get_db)):
    if modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O modelo de Machine Learning não está disponível no servidor",
        )

    # Engenharia de atributos (idêntica ao treinar.py)
    totalOcupacao = (
        dados.ocupacaoCovidUti
        + dados.ocupacaoCovidCli
        + dados.ocupacaoHospitalarUti
        + dados.ocupacaoHospitalarCli
    )
    proporcaoCovidUti = dados.ocupacaoCovidUti / (totalOcupacao + 1e-5)
    razaoAltasOcupacao = dados.saidaConfirmadaAltas / (totalOcupacao + 1e-5)

    # Mesma ordem de colunas do treino
    features = pd.DataFrame([{
        "ocupacaoCovidUti": dados.ocupacaoCovidUti,
        "ocupacaoCovidCli": dados.ocupacaoCovidCli,
        "ocupacaoHospitalarUti": dados.ocupacaoHospitalarUti,
        "ocupacaoHospitalarCli": dados.ocupacaoHospitalarCli,
        "saidaConfirmadaAltas": dados.saidaConfirmadaAltas,
        "totalOcupacao": totalOcupacao,
        "proporcaoCovidUti": proporcaoCovidUti,
        "razaoAltasOcupacao": razaoAltasOcupacao,
    }])

    predicao = int(modelo.predict(features)[0])

    if hasattr(modelo, "predict_proba"):
        prob_obito = float(modelo.predict_proba(features)[0][1] * 100)
    else:
        prob_obito = 100.0 if predicao == 1 else 0.0

    novo_registro = HistoricoPrevisaoModel(
        ocupacaoCovidUti=dados.ocupacaoCovidUti,
        ocupacaoCovidCli=dados.ocupacaoCovidCli,
        ocupacaoHospitalarUti=dados.ocupacaoHospitalarUti,
        ocupacaoHospitalarCli=dados.ocupacaoHospitalarCli,
        saidaConfirmadaAltas=dados.saidaConfirmadaAltas,
        totalOcupacao=totalOcupacao,
        predicao=predicao,
        probabilidade_obito=round(prob_obito, 2),
    )
    db.add(novo_registro)
    db.commit()
    db.refresh(novo_registro)

    mensagem_status = (
        "ALERTA CRÍTICO: Risco elevado de óbito detectado. Recomenda-se acompanhamento intensivo."
        if predicao == 1
        else "SITUAÇÃO ESTÁVEL: Parâmetros hospitalares dentro dos limites de baixo risco."
    )

    return PrevisaoResposta(
        id_simulacao=novo_registro.id,
        data_hora=novo_registro.data_hora,
        risco_obito_detectado=predicao,
        probabilidade_obito_pct=round(prob_obito, 2),
        total_ocupacao=totalOcupacao,
        proporcao_covid_uti_pct=round(proporcaoCovidUti * 100, 2),
        mensagem=mensagem_status,
    )


@app.get(
    "/historico",
    response_model=List[HistoricoItem],
    tags=["Histórico & Auditoria"],
    summary="Consulta as últimas simulações salvas no banco",
)
def listar_historico(limit: int = 50, db: Session = Depends(get_db)):
    """Registros do SQLite, do mais recente para o mais antigo."""
    return (
        db.query(HistoricoPrevisaoModel)
        .order_by(HistoricoPrevisaoModel.id.desc())
        .limit(limit)
        .all()
    )