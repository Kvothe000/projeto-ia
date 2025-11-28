# Binance/gerar_dataset_50.py - VERSÃO FINAL (NORMALIZADA)
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
from indicators import Calculadora
import time

QTD_MOEDAS = 50
TIMEFRAME = "15m"
QTD_POR_MOEDA = 12000 
ALVO_LUCRO = 0.006     # 0.6% (Meta Scalper)
ALVO_STOP = 0.003      # 0.3%
FUTURO_VISAO = 4

def obter_top_50_moedas(connector):
    print("🛰️ Buscando Top 50 moedas...")
    try:
        tickers = connector.client.futures_ticker()
        df = pd.DataFrame(tickers)
        df = df[df['symbol'].str.endswith('USDT')]
        df['quoteVolume'] = pd.to_numeric(df['quoteVolume'])
        return df.sort_values('quoteVolume', ascending=False).head(QTD_MOEDAS)['symbol'].tolist()
    except: return []

def buscar_historico_rapido(connector, par):
    try:
        klines = []
        end_time = None
        for _ in range(8): # 8 blocos de 1500 = 12.000 candles
            k = connector.client.futures_klines(symbol=par, interval=TIMEFRAME, limit=1500, endTime=end_time)
            if not k: break
            klines.extend(k)
            end_time = int(k[0][0]) - 1
            time.sleep(0.1)
        klines.sort(key=lambda x: x[0])
        return connector._tratar_df(klines)
    except: return None

def processar_dados(df):
    df = Calculadora.adicionar_todos(df)
    
    # VWAP Acumulada
    df['pv'] = df['close'] * df['volume']
    df['VWAP'] = df['pv'].cumsum() / df['volume'].cumsum()
    df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
    
    # --- NORMALIZAÇÃO (O SEGREDO) ---
    vol_media = df['volume'].rolling(20).mean()
    df['CVD_Slope'] = df['CVD'].diff(3) / vol_media # Normaliza fluxo
    df['Vol_Relativo'] = df['volume'] / vol_media
    df['ATRr'] = df['ATRr_14']
    
    # Targets
    targets = []
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(len(df)):
        if i + FUTURO_VISAO >= len(df): targets.append(0); continue
        
        entry = closes[i]
        # 1=Long, 2=Short
        res = 0
        # Checa Long
        for j in range(1, FUTURO_VISAO + 1):
            if lows[i+j] <= entry*(1-ALVO_STOP): break
            if highs[i+j] >= entry*(1+ALVO_LUCRO): res = 1; break
        
        # Se não deu Long, Checa Short
        if res == 0:
            for j in range(1, FUTURO_VISAO + 1):
                if highs[i+j] >= entry*(1+ALVO_STOP): break
                if lows[i+j] <= entry*(1-ALVO_LUCRO): res = 2; break
                
        targets.append(res)
    
    df['target'] = targets
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    return df.iloc[:-FUTURO_VISAO]

def main():
    con = BinanceConnector()
    moedas = obter_top_50_moedas(con)
    dfs = []
    
    print(f"🚜 Iniciando Mineração Massiva ({len(moedas)} moedas)...")
    for i, par in enumerate(moedas):
        print(f"[{i+1}/{len(moedas)}] {par}...", end="\r")
        df_raw = buscar_historico_rapido(con, par)
        if df_raw is not None and len(df_raw) > 1000:
            df_proc = processar_dados(df_raw)
            # Colunas Finais
            cols = ['RSI_14', 'ADX_14', 'Dist_VWAP', 'CVD_Slope', 'Vol_Relativo', 'ATRr', 'target']
            dfs.append(df_proc[cols])
            
    if dfs:
        print("\n🌪️ Unificando dados...")
        df_final = pd.concat(dfs, ignore_index=True)
        df_final = df_final.sample(frac=1).reset_index(drop=True)
        df_final.to_csv("dataset_50_coins_norm.csv", index=False)
        print(f"\n💾 Dataset Final: {len(df_final)} linhas. PRONTO PARA TREINO!")

if __name__ == "__main__":
    main()