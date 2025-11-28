# Binance/gerar_dataset_v7.py - TIME SERIES STRICT + NOVAS FEATURES
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
        return df.sort_values('quoteVolume', ascending=False).head(QTD_MOEDAS)['symbol'].tolist()
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
    df_moeda = df_moeda.set_index('timestamp')
    df_btc_indexed = df_btc.set_index('timestamp')[['close', 'volume']]
    df_btc_indexed.columns = ['btc_close', 'btc_volume']
    df_merged = df_moeda.join(df_btc_indexed, how='inner')
    return df_merged.reset_index()

def processar_dados(df):
    df = Calculadora.adicionar_todos(df)
    
    # --- FEATURES CLÁSSICAS ---
    df['pv'] = df['close'] * df['volume']
    df['VWAP'] = df['pv'].cumsum() / df['volume'].cumsum()
    df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
    
    # --- FEATURES NOVAS (V7) ---
    
    # 1. Z-Score de Volume (Anomalia Estatística)
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['Vol_ZScore'] = (df['volume'] - vol_mean) / vol_std
    
    # 2. CVD Normalizado
    df['CVD_Slope'] = df['CVD'].diff(3) / vol_mean
    
    # 3. Contexto BTC
    df['BTC_Change'] = df['btc_close'].pct_change(3)
    df['Rel_Strength'] = df['close'].pct_change(3) - df['BTC_Change']
    
    # 4. Features Temporais (Ciclos)
    # Transforma timestamp em hora (0-23) e dia da semana (0-6)
    dates = pd.to_datetime(df['timestamp'], unit='ms')
    df['Hour_Sin'] = np.sin(2 * np.pi * dates.dt.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * dates.dt.hour / 24)
    # A IA vai aprender que "Tal hora costuma ter reversão"
    
    df['ATRr'] = df['ATRr_14']
    
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
    
    print("👑 Baixando BTC Mestre...")
    df_btc = buscar_historico_rapido(con, "BTCUSDT")
    if df_btc is None: return

    moedas = obter_top_50_moedas(con)
    dfs = []
    
    print(f"🚜 Minerando V7 (Features Temporais + ZScore)...")
    
    for i, par in enumerate(moedas):
        if par == "BTCUSDT": continue
        print(f"[{i+1}/{len(moedas)}] {par}...", end="\r")
        
        df_raw = buscar_historico_rapido(con, par)
        if df_raw is not None and len(df_raw) > 1000:
            df_merged = adicionar_contexto_btc(df_raw, df_btc)
            df_proc = processar_dados(df_merged)
            
            # Adicionamos as novas colunas
            cols = [
                'RSI_14', 'ADX_14', 'Dist_VWAP', 'CVD_Slope', 
                'Vol_ZScore', 'ATRr', 'BTC_Change', 'Rel_Strength',
                'Hour_Sin', 'Hour_Cos', # <--- O Tempo é chave
                'target', 'timestamp'   # Timestamp para ordenar depois
            ]
            dfs.append(df_proc[cols])
            
    if dfs:
        print("\n🌪️ Consolidando sem embaralhar (Time Series)...")
        df_final = pd.concat(dfs, ignore_index=True)
        
        # ORDENA POR TEMPO (CRUCIAL PARA VALIDAÇÃO REAL)
        df_final = df_final.sort_values('timestamp').reset_index(drop=True)
        # Remove o timestamp antes de salvar, pois a IA não deve ler o número bruto
        df_final.drop(columns=['timestamp'], inplace=True)
        
        df_final.to_csv("dataset_v7_timeseries.csv", index=False)
        print(f"\n💾 DATASET V7 GERADO! {len(df_final)} linhas ordenadas cronologicamente.")

if __name__ == "__main__":
    main()