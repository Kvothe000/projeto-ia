# Binance/dashboard.py (VERSÃO VISUAL COMPLETA)
import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="Predador V5 Visual", page_icon="🦅", layout="wide", initial_sidebar_state="collapsed")
st.title("🦅 Predador V5 - Centro de Inteligência Visual")

# --- FUNÇÕES ---
def carregar_csv(arquivo, linhas=100):
    if os.path.exists(arquivo):
        try:
            # Lê apenas as últimas N linhas para ser rápido
            return pd.read_csv(arquivo).tail(linhas)
        except: pass
    return pd.DataFrame()

def carregar_live():
    if os.path.exists("monitor_live.json"):
        try:
            with open("monitor_live.json", 'r') as f: return json.load(f)
        except: pass
    return None

# Dados
df_trades = carregar_csv("trades_history.csv", 200)
df_analises = carregar_csv("analysis_history.csv", 300) # Últimas 300 análises
dados_live = carregar_live()

# Auto-Refresh
if st.button('🔄 Atualizar'): st.rerun()

# ==================================================
# 1. RADAR EM TEMPO REAL (TOPO)
# ==================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("📡 Radar Ao Vivo")
    if dados_live and "moedas" in dados_live:
        df_live = pd.DataFrame(dados_live["moedas"])
        st.dataframe(
            df_live.style.applymap(lambda x: 'color: #00FF00' if x=='BUY' else ('color: #FF0000' if x=='SELL' else ''), subset=['sinal']),
            use_container_width=True,
            column_config={
                "confianca": st.column_config.NumberColumn("Confiança", format="%.1f%%"),
                "adx": st.column_config.ProgressColumn("ADX", min_value=0, max_value=60, format="%.0f"),
            }
        )
    else:
        st.info("A aguardar dados do Scanner...")

with col2:
    st.subheader("🎯 Status")
    n_trades = len(df_trades) if not df_trades.empty else 0
    vol = df_trades['valor_usdt'].sum() if not df_trades.empty else 0
    st.metric("Trades Hoje", n_trades)
    st.metric("Volume ($)", f"${vol:,.0f}")
    if not df_analises.empty:
        last_conf = df_analises.iloc[-1]['confianca']
        st.metric("Humor da IA", f"{last_conf}%")

st.markdown("---")

# ==================================================
# 2. GRÁFICOS DE INTELIGÊNCIA (NOVO!)
# ==================================================
st.subheader("🧠 Cérebro da IA (Visualização Temporal)")

if not df_analises.empty:
    # Filtro por moeda para o gráfico não ficar bagunçado
    moedas_disp = df_analises['par'].unique()
    moeda_selecionada = st.selectbox("Selecione a Moeda para visualizar:", moedas_disp, index=0)
    
    # Filtra dados da moeda
    df_chart = df_analises[df_analises['par'] == moeda_selecionada].copy()
    
    # Cria Gráfico Combinado (Preço vs Confiança)
    fig = go.Figure()

    # Linha de Preço (Eixo Y Esquerdo)
    fig.add_trace(go.Scatter(
        x=df_chart['data'], y=df_chart['preco'],
        name='Preço', line=dict(color='white', width=1),
        yaxis='y1'
    ))

    # Linha de Confiança (Eixo Y Direito)
    # Pinta de Verde se for BUY, Vermelho se for SELL
    colors = ['green' if s == 'BUY' else ('red' if s == 'SELL' else 'gray') for s in df_chart['sinal']]
    
    fig.add_trace(go.Scatter(
        x=df_chart['data'], y=df_chart['confianca'],
        name='Confiança IA %', mode='markers+lines',
        marker=dict(color=colors, size=6),
        line=dict(color='rgba(255, 255, 255, 0.2)', width=1, dash='dot'),
        yaxis='y2'
    ))

    # Layout com Eixo Duplo
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Preço ($)", side="left", showgrid=False),
        yaxis2=dict(title="Confiança IA (%)", side="right", overlaying="y", showgrid=True, range=[0, 100]),
        legend=dict(orientation="h", y=1.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Expander para ver os dados brutos da análise
    with st.expander(f"Ver Log Detalhado de Análises ({len(df_chart)} registros)"):
        st.dataframe(df_chart.sort_index(ascending=False), use_container_width=True)

else:
    st.warning("Aguardando dados de análise para gerar gráficos...")

st.markdown("---")

# ==================================================
# 3. HISTÓRICO DE TRADES (O DINHEIRO)
# ==================================================
st.subheader("💰 Histórico de Execuções (Trades)")

if not df_trades.empty:
    st.dataframe(
        df_trades.sort_index(ascending=False).head(100),
        use_container_width=True,
        column_config={
            "lado": st.column_config.TextColumn("Lado"),
            "resultado": st.column_config.TextColumn("Status"),
            "valor_usdt": st.column_config.NumberColumn("Valor", format="$%.2f")
        }
    )
else:
    st.text("Nenhum trade executado ainda.")

# Refresh Automático
time.sleep(2)
st.rerun()