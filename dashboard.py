# Binance/dashboard.py (DASHBOARD FINANCEIRO)
import streamlit as st
import pandas as pd
import time
import json
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Gênesis Pro", page_icon="🦅", layout="wide")
st.title("🦅 Gênesis AI - Painel de Controle")

# Carrega Dados
def get_data():
    monitor, wallet, history = {}, {"saldo": 200}, pd.DataFrame()
    
    if os.path.exists("monitor_live.json"):
        try: monitor = json.load(open("monitor_live.json"))
        except: pass
        
    if os.path.exists("bot_wallet.json"):
        try: wallet = json.load(open("bot_wallet.json"))
        except: pass
        
    if os.path.exists("trades_history.csv"):
        try: history = pd.read_csv("trades_history.csv")
        except: pass
        
    return monitor, wallet, history

monitor, wallet, df_hist = get_data()

# --- KPI PRINCIPAL ---
saldo_atual = wallet.get("saldo", 200.0)
saldo_inicial = wallet.get("saldo_inicial", 200.0)
lucro_total_usd = saldo_atual - saldo_inicial
lucro_total_pct = (lucro_total_usd / saldo_inicial) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 Saldo Atual", f"${saldo_atual:.2f}", f"{lucro_total_pct:+.2f}%")

with col2:
    if not df_hist.empty:
        last_trade = df_hist.iloc[-1]
        pnl = last_trade.get('pnl_pct', 0)
        if pd.isna(pnl): pnl = 0
        st.metric("Último Trade", f"{last_trade['par']}", f"{pnl:+.2f}%")
    else:
        st.metric("Último Trade", "---", "0%")

with col3:
    status = "AGUARDANDO"
    if monitor and "moedas" in monitor:
        status = monitor["moedas"][0]["sinal"]
    st.metric("Sinal IA", status)

with col4:
    trades_hoje = len(df_hist) if not df_hist.empty else 0
    st.metric("Trades Totais", trades_hoje)

st.markdown("---")

# --- GRÁFICOS E TABELAS ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 Crescimento da Conta")
    if not df_hist.empty:
        # Recria curva de saldo baseada no histórico
        # Começa com saldo inicial e soma os PnLs
        df_hist['saldo_acumulado'] = saldo_inicial + df_hist['pnl_usd'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist['data'], y=df_hist['saldo_acumulado'],
            mode='lines+markers', name='Saldo',
            line=dict(color='#00ff00', width=2)
        ))
        fig.update_layout(
            template="plotly_dark", 
            height=350,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Evolução do Patrimônio ($)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aguardando primeiro trade fechado para gerar gráfico.")

with c2:
    st.subheader("📡 Radar Ao Vivo")
    if monitor and "moedas" in monitor:
        st.dataframe(
            pd.DataFrame(monitor["moedas"]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.text("Scanner offline...")

# Histórico Recente
st.subheader("📝 Últimas Operações")
if not df_hist.empty:
    st.dataframe(
        df_hist.sort_index(ascending=False).head(10),
        use_container_width=True
    )

# Auto-Refresh
time.sleep(2)
st.rerun()