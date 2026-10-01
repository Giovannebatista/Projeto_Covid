from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# permite que o HTML faça requisições para a API local sem bloqueio do navegador
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo de IA
modelo = joblib.load('Modelo_Covid.pkl')

# Define a estrutura de dados esperado do formulário
class DadosEntrada(BaseModel):
    ocupacaoCovidUti: float
    ocupacaoCovidCli: float
    ocupacaoHospitalarUti: float
    ocupacaoHospitalarCli: float
    saidaConfirmadaAltas: float

@app.post("/prever")
def prever(dados: DadosEntrada):
    # Formata a entrada para o padrão do scikit-learn
    entrada = [[
        dados.ocupacaoCovidUti,
        dados.ocupacaoCovidCli,
        dados.ocupacaoHospitalarUti,
        dados.ocupacaoHospitalarCli,
        dados.saidaConfirmadaAltas
    ]]

    # Realiza a previsão (0 = sem óbito, 1 = com óbito)
    resultado = modelo.predict(entrada)[0]

    mensagem = "Alto risco de ocorrência de óbito" if resultado == 1 else "Baixo risco de ocorrência de óbito"

    return {"previsao": mensagem, "classe": int(resultado)}