# Binance/dashboard.py (VISUALIZAÇÃO COMPLETA)
import streamlit as st
import pandas as pd
import time
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# Configuração
st.set_page_config(page_title="Gênesis Pro", page_icon="🦅", layout="wide")
st.title("🦅 Gênesis AI - Centro de Comando")

# Função de Carregamento Seguro
def get_data():
    monitor = {}
    wallet = {"saldo": 200, "saldo_inicial": 200}
    history = pd.DataFrame()
    
    # Tenta ler arquivos JSON/CSV com tolerância a falhas de leitura/escrita
    try:
        if os.path.exists("monitor_live.json"):
            with open("monitor_live.json", "r") as f: monitor = json.load(f)
    except: pass
        
    try:
        if os.path.exists("bot_wallet.json"):
            with open("bot_wallet.json", "r") as f: wallet = json.load(f)
    except: pass
        
    try:
        if os.path.exists("trades_history.csv"):
            history = pd.read_csv("trades_history.csv")
    except: pass
        
    return monitor, wallet, history

# Loop de Dados
monitor, wallet, df_hist = get_data()

# --- 1. MÉTRICAS FINANCEIRAS (TOPO) ---
saldo_atual = wallet.get("saldo", 200.0)
saldo_em_uso = wallet.get("em_uso", 0.0)
saldo_total = saldo_atual + saldo_em_uso
saldo_inicial = wallet.get("saldo_inicial", 200.0)

lucro_total = saldo_total - saldo_inicial
lucro_pct = (lucro_total / saldo_inicial) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 Patrimônio Total", f"${saldo_total:.2f}", f"{lucro_pct:+.2f}%")

with col2:
    st.metric("🛡️ Capital Livre", f"${saldo_atual:.2f}")

with col3:
    # Pega status da WLD
    status_wld = "OFFLINE"
    preco_wld = 0
    if monitor and "moedas" in monitor:
        for m in monitor["moedas"]:
            if "WLD" in m["par"]:
                status_wld = m["sinal"]
                preco_wld = m["preco"]
    st.metric("Sinal WLD", status_wld, f"${preco_wld}")

with col4:
    # Trades Fechados
    if not df_hist.empty:
        trades_fechados = df_hist[df_hist['tipo'].isin(['CLOSE', 'Trailing Stop', 'IA (Close)', 'Inversão'])].shape[0]
    else:
        trades_fechados = 0
    st.metric("Trades Finalizados", trades_fechados)

st.markdown("---")

# --- 2. GRÁFICO DE PERFORMANCE (EQUITY CURVE) ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 Curva de Crescimento")
    
    if not df_hist.empty:
        # Filtra apenas linhas que alteram o saldo (CLOSE ou SELL de fechamento)
        # Simplificação: Usamos todas as linhas que têm PnL registrado
        df_chart = df_hist[df_hist['pnl_usd'] != 0].copy()
        
        if not df_chart.empty:
            # Cria a curva acumulada
            df_chart['saldo_acumulado'] = saldo_inicial + df_chart['pnl_usd'].cumsum()
            
            # Adiciona ponto inicial (dia 0)
            inicio = pd.DataFrame({'data': [df_chart.iloc[0]['data']], 'saldo_acumulado': [saldo_inicial]})
            # df_chart = pd.concat([inicio, df_chart]) # Opcional
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_chart['data'], 
                y=df_chart['saldo_acumulado'],
                mode='lines+markers',
                name='Equity',
                line=dict(color='#00ff00', width=3),
                fill='tozeroy', # Efeito visual bonito
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))
            
            fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                title="Evolução do Saldo ($)",
                xaxis_title="Tempo",
                yaxis_title="Capital"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aguardando primeiro lucro realizado para gerar gráfico...")
    else:
        st.info("Nenhum histórico encontrado.")

with c2:
    st.subheader("📡 Radar Ao Vivo")
    if monitor and "moedas" in monitor:
        st.dataframe(
            pd.DataFrame(monitor["moedas"]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Scanner/Trader não detectado.")

# --- 3. HISTÓRICO DETALHADO (TABELA) ---
st.subheader("📝 Histórico de Operações")

if not df_hist.empty:
    # Formatação bonita
    df_show = df_hist.sort_index(ascending=False).copy()
    
    st.dataframe(
        df_show,
        use_container_width=True,
        column_config={
            "pnl_usd": st.column_config.NumberColumn("Lucro ($)", format="$%.2f"),
            "pnl_pct": st.column_config.NumberColumn("Lucro (%)", format="%.2f%%"),
            "preco": st.column_config.NumberColumn("Preço", format="$%.4f"),
            "valor_usdt": st.column_config.NumberColumn("Volume", format="$%.2f"),
        }
    )

# Auto-Refresh a cada 2s
time.sleep(2)
st.rerun()