import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Endereço da API (troque via variável de ambiente ao fazer deploy)
API_URL = os.getenv("API_URL", "http://localhost:8000")
# URL da API vista pelo navegador (dentro do Docker, API_URL é http://api:8000,
# que só existe na rede interna dos contêineres)
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", API_URL)

st.set_page_config(page_title="Simulador Preditivo COVID-19", page_icon="🏥", layout="wide")


# ---------------------------------------------------------------
# Funções de comunicação com a API (único ponto de contato com o backend)
# ---------------------------------------------------------------
@st.cache_data(ttl=5)
def api_online() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.ok and r.json().get("modelo_carregado", False)
    except requests.RequestException:
        return False


def chamar_predict(payload: dict) -> dict:
    """POST /predict. Levanta RuntimeError com mensagem amigável em caso de falha."""
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.ConnectionError:
        raise RuntimeError(f"Não foi possível conectar à API em {API_URL}.")
    except requests.Timeout:
        raise RuntimeError("A API demorou demais para responder.")

    if r.status_code == 422:
        raise RuntimeError("A API rejeitou os dados enviados (validação).")
    if r.status_code == 503:
        raise RuntimeError("A API está no ar, mas o modelo não foi carregado.")
    if not r.ok:
        raise RuntimeError(f"Erro {r.status_code} da API: {r.text}")
    return r.json()


def buscar_historico(limit: int = 10) -> pd.DataFrame:
    """GET /historico: o histórico agora vem do banco, não do session_state."""
    try:
        r = requests.get(f"{API_URL}/historico", params={"limit": limit}, timeout=5)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except requests.RequestException:
        return pd.DataFrame()


# ---------------------------------------------------------------
# Verificação da API
# ---------------------------------------------------------------
if not api_online():
    st.error(
        f"❌ API indisponível em **{API_URL}**.\n\n"
        "Inicie o backend com: `uvicorn main:app --reload --port 8000`"
    )
    if st.button("🔄 Tentar novamente"):
        api_online.clear()
        st.rerun()
    st.stop()

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/coronavirus.png", width=70)
    st.title("Sobre o Modelo")
    st.markdown("""
    **Algoritmo:** Random Forest Classifier  
    **Técnica de Balanceamento:** SMOTE  
    **Versão da API:** `v2.0.0`  

    ---
    ### 📌 Variáveis mais Relevantes:
    1. Ocupação de UTI COVID
    2. Proporção de UTI em relação ao Total
    3. Razão de Altas por Ocupação
    """)
    st.divider()
    st.success("🟢 API conectada")
    st.markdown(f"[📖 Swagger]({API_PUBLIC_URL}/docs)")
    st.caption("Desenvolvido para apoio à gestão de leitos hospitalares.")

# ---------------------------------------------------------------
# Cabeçalho e cenários
# ---------------------------------------------------------------
st.title("🏥 Simulador Preditivo de Risco Hospitalar")
st.markdown("Ajuste os parâmetros ou selecione um **cenário rápido** para calcular a probabilidade de risco.")

st.subheader("⚡ Carregar Cenário de Teste")
cenario = st.selectbox(
    "Escolha uma simulação rápida:",
    ["Personalizado", "🔴 Cenário Crítico (Saturação)", "🟡 Cenário Moderado", "🟢 Cenário Estável (Baixa Ocupação)"],
)

defaults = {
    "Personalizado": (5, 10, 15, 20, 2),
    "🔴 Cenário Crítico (Saturação)": (45, 30, 20, 10, 1),
    "🟡 Cenário Moderado": (15, 20, 10, 15, 5),
    "🟢 Cenário Estável (Baixa Ocupação)": (2, 5, 5, 10, 8),
}
val_uti_covid, val_cli_covid, val_uti_hosp, val_cli_hosp, val_altas = defaults[cenario]

st.divider()

# ---------------------------------------------------------------
# Formulário
# ---------------------------------------------------------------
st.subheader("📋 Parâmetros de Ocupação do Hospital")
col_in1, col_in2 = st.columns(2)

# A key inclui o cenário para os campos atualizarem ao trocar o preset
with col_in1:
    uti_covid = st.number_input("Ocupação UTI COVID:", min_value=0, value=val_uti_covid, step=1, key=f"uc_{cenario}")
    cli_covid = st.number_input("Ocupação Enfermaria COVID:", min_value=0, value=val_cli_covid, step=1, key=f"cc_{cenario}")
    uti_hosp = st.number_input("Ocupação UTI Geral:", min_value=0, value=val_uti_hosp, step=1, key=f"uh_{cenario}")
with col_in2:
    cli_hosp = st.number_input("Ocupação Enfermaria Geral:", min_value=0, value=val_cli_hosp, step=1, key=f"ch_{cenario}")
    altas = st.number_input("Saídas por Alta Confirmada:", min_value=0, value=val_altas, step=1, key=f"al_{cenario}")

st.divider()

# ---------------------------------------------------------------
# Processamento via API
# ---------------------------------------------------------------
if st.button("🚀 Processar Análise com IA", type="primary", use_container_width=True):
    payload = {
        "ocupacaoCovidUti": float(uti_covid),
        "ocupacaoCovidCli": float(cli_covid),
        "ocupacaoHospitalarUti": float(uti_hosp),
        "ocupacaoHospitalarCli": float(cli_hosp),
        "saidaConfirmadaAltas": float(altas),
    }

    try:
        with st.spinner("Consultando a API..."):
            resp = chamar_predict(payload)
    except RuntimeError as e:
        st.error(f"⚠️ {e}")
    else:
        alto_risco = resp["risco_obito_detectado"] == 1
        prob = resp["probabilidade_obito_pct"]

        col_res1, col_res2 = st.columns([1.2, 1])

        with col_res1:
            st.subheader("📊 Diagnóstico da IA")
            m1, m2 = st.columns(2)
            m1.metric("Ocupação Total", f"{resp['total_ocupacao']:.0f} leitos")
            m2.metric("Proporção UTI COVID", f"{resp['proporcao_covid_uti_pct']:.1f}%")

            if alto_risco:
                st.error(f"⚠️ **{resp['mensagem']}**\n\nProbabilidade de risco: **{prob:.1f}%**")
                st.warning("""
                **Recomendações Operacionais:**
                - Mobilizar leitos de reserva ou planejar transferência de pacientes.
                - Priorizar triagem rigorosa na admissão de UTI.
                - Avaliar o fluxo de altas da enfermaria para otimizar rotação.
                """)
            else:
                st.success(f"✅ **{resp['mensagem']}**\n\nProbabilidade de risco: **{prob:.1f}%**")
                st.info("""
                **Recomendações Operacionais:**
                - Capacidade hospitalar operando dentro dos limites de segurança.
                - Manter monitorização contínua das altas e internações diárias.
                """)
            st.caption(f"Simulação #{resp['id_simulacao']} registrada no banco de dados.")

        with col_res2:
            fig_donut = go.Figure(data=[go.Pie(
                labels=["Risco (%)", "Segurança (%)"],
                values=[prob, max(0.0, 100.0 - prob)],
                hole=0.6,
                marker_colors=["#e74c3c" if prob >= 50 else "#f39c12", "#2ecc71"],
            )])
            fig_donut.update_layout(
                title_text="Nível de Risco Calculado",
                showlegend=True,
                height=320,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

# ---------------------------------------------------------------
# Histórico (vem do banco via GET /historico)
# ---------------------------------------------------------------
df_hist = buscar_historico(limit=10)
if not df_hist.empty:
    st.divider()
    st.subheader("📜 Últimas Simulações (banco de dados)")
    df_hist["data_hora"] = pd.to_datetime(df_hist["data_hora"]).dt.strftime("%d/%m/%Y %H:%M:%S")
    df_hist["Status"] = df_hist["predicao"].map({1: "Alto Risco", 0: "Baixo Risco"})
    df_hist["Risco (%)"] = df_hist["probabilidade_obito"].round(1)
    st.dataframe(
        df_hist[["id", "data_hora", "ocupacaoCovidUti", "totalOcupacao",
                 "saidaConfirmadaAltas", "Risco (%)", "Status"]]
        .rename(columns={"data_hora": "Data/Hora", "ocupacaoCovidUti": "UTI COVID",
                         "totalOcupacao": "Total Ocupação", "saidaConfirmadaAltas": "Altas"}),
        use_container_width=True,
        hide_index=True,
    )