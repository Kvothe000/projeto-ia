# Binance/gerar_dataset_v6.py
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
from indicators import Calculadora
import time

# --- CONFIGURAÇÃO ---
QTD_MOEDAS = 50        
TIMEFRAME = "15m"
QTD_POR_MOEDA = 12000  
ALVO_LUCRO = 0.006     
ALVO_STOP = 0.003      
FUTURO_VISAO = 4

def obter_top_50_moedas(connector):
    print("🛰️ Selecionando as Top 50 moedas...")
    try:
        tickers = connector.client.futures_ticker()
        df = pd.DataFrame(tickers)
        df = df[df['symbol'].str.endswith('USDT')]
        df['quoteVolume'] = pd.to_numeric(df['quoteVolume'])
        top_50 = df.sort_values('quoteVolume', ascending=False).head(QTD_MOEDAS)
        return top_50['symbol'].tolist()
    except: return []

def buscar_historico_rapido(connector, par):
    try:
        klines = []
        end_time = None
        for _ in range(8):
            k = connector.client.futures_klines(
                symbol=par, interval=TIMEFRAME, limit=1500, endTime=end_time
            )
            if not k: break
            klines.extend(k)
            end_time = int(k[0][0]) - 1
            time.sleep(0.1)
        klines.sort(key=lambda x: x[0])
        return connector._tratar_df(klines)
    except: return None

def adicionar_contexto_btc(df_moeda, df_btc):
    # Garante que timestamps são índices para cruzamento rápido
    df_moeda = df_moeda.set_index('timestamp')
    df_btc_indexed = df_btc.set_index('timestamp')[['close', 'volume']]
    df_btc_indexed.columns = ['btc_close', 'btc_volume']
    
    # Junta (Inner Join) - Só mantém candles onde temos dados de ambos
    df_merged = df_moeda.join(df_btc_indexed, how='inner')
    return df_merged.reset_index()

def processar_dados(df):
    df = Calculadora.adicionar_todos(df)
    
    # Indicadores da Moeda
    df['pv'] = df['close'] * df['volume']
    df['VWAP'] = df['pv'].cumsum() / df['volume'].cumsum()
    df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
    
    vol_media = df['volume'].rolling(20).mean()
    df['CVD_Slope'] = df['CVD'].diff(3) / vol_media
    df['Vol_Relativo'] = df['volume'] / vol_media
    df['ATRr'] = df['ATRr_14']
    
    # --- CONTEXTO BTC (O Segredo da V6) ---
    df['BTC_Change'] = df['btc_close'].pct_change(3) # O que o BTC fez nos últimos 45min?
    # Força Relativa: A moeda subiu mais que o BTC?
    df['Rel_Strength'] = df['close'].pct_change(3) - df['BTC_Change']
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    
    # Alvos
    targets = []
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(len(df)):
        if i + FUTURO_VISAO >= len(df):
            targets.append(0); continue
            
        entry = closes[i]
        tp_long, sl_long = entry * (1 + ALVO_LUCRO), entry * (1 - ALVO_STOP)
        tp_short, sl_short = entry * (1 - ALVO_LUCRO), entry * (1 + ALVO_STOP)
        
        res = 0
        for j in range(1, FUTURO_VISAO + 1):
            if lows[i+j] <= sl_long: break
            if highs[i+j] >= tp_long: res = 1; break
            
        if res == 0:
            for j in range(1, FUTURO_VISAO + 1):
                if highs[i+j] >= sl_short: break
                if lows[i+j] <= tp_short: res = 2; break
                
        targets.append(res)
    
    df['target'] = targets
    return df.iloc[:-FUTURO_VISAO]

def main():
    con = BinanceConnector()
    
    print("👑 Baixando histórico mestre do BITCOIN...")
    df_btc = buscar_historico_rapido(con, "BTCUSDT")
    if df_btc is None: return

    moedas = obter_top_50_moedas(con)
    dfs = []
    
    print(f"🚜 Iniciando Mineração V6 com Contexto...")
    
    for i, par in enumerate(moedas):
        if par == "BTCUSDT": continue
        print(f"[{i+1}/{len(moedas)}] {par}...", end="\r")
        
        df_raw = buscar_historico_rapido(con, par)
        if df_raw is not None and len(df_raw) > 1000:
            # Adiciona o BTC ao lado da moeda
            df_merged = adicionar_contexto_btc(df_raw, df_btc)
            df_proc = processar_dados(df_merged)
            
            cols = [
                'RSI_14', 'ADX_14', 'Dist_VWAP', 'CVD_Slope', 
                'Vol_Relativo', 'ATRr', 'BTC_Change', 'Rel_Strength', 
                'target'
            ]
            dfs.append(df_proc[cols])
            
    if dfs:
        print("\n🌪️ Consolidando Inteligência V6...")
        df_final = pd.concat(dfs, ignore_index=True)
        df_final = df_final.sample(frac=1).reset_index(drop=True)
        
        df_final.to_csv("dataset_v6_context.csv", index=False)
        print(f"\n💾 SUCESSO! dataset_v6_context.csv gerado.")

if __name__ == "__main__":
    main()