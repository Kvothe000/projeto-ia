# Binance/gerar_dataset_v11_fusion.py (CORRIGIDO: INCLUI PREÇO)
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
import pandas_ta as ta
import time
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Genesis_AI'))
from features_engine import FeaturesEngine

# --- CONFIGURAÇÃO V11 ---
QTD_MOEDAS = 50
TIMEFRAME = "15m"
QTD_POR_MOEDA = 6000   # 6000 candles = ~2 meses
HORIZONTE_ALVO = 8     
ALVO_LUCRO = 0.008     

def obter_top_50_moedas(connector):
    try:
        tickers = connector.client.futures_ticker()
        df = pd.DataFrame(tickers)
        df = df[df['symbol'].str.endswith('USDT')]
        df['quoteVolume'] = pd.to_numeric(df['quoteVolume'])
        return df.sort_values('quoteVolume', ascending=False).head(QTD_MOEDAS)['symbol'].tolist()
    except: return []

def buscar_historico(connector, par):
    try:
        klines = []
        end_time = None
        for _ in range(5): 
            k = connector.client.futures_klines(symbol=par, interval=TIMEFRAME, limit=1500, endTime=end_time)
            if not k: break
            klines.extend(k)
            end_time = int(k[0][0]) - 1
            time.sleep(0.1)
        klines.sort(key=lambda x: x[0])
        return connector._tratar_df(klines)
    except: return None

def processar_fusao(df_moeda, df_btc):
    # 1. Merge Inicial
    df_moeda = df_moeda.set_index('timestamp')
    df_btc = df_btc.set_index('timestamp')[['close', 'volume']]
    df_btc.columns = ['btc_close', 'btc_volume']
    
    df = df_moeda.join(df_btc, how='inner').reset_index()
    
    for c in ['close', 'high', 'low', 'volume', 'btc_close', 'btc_volume']:
        df[c] = df[c].astype(float)

    # 2. Features Matemáticas
    for p in [3, 5, 10, 20]:
        df[f'mom_{p}'] = df['close'].pct_change(p)

    vol_curta = df['close'].pct_change().rolling(5).std()
    vol_longa = df['close'].pct_change().rolling(20).std()
    df['vol_ratio'] = vol_curta / vol_longa

    max_20 = df['high'].rolling(20).max()
    min_20 = df['low'].rolling(20).min()
    df['pos_canal'] = (df['close'] - min_20) / (max_20 - min_20)

    ema9 = df.ta.ema(close=df['close'], length=9)
    ema21 = df.ta.ema(close=df['close'], length=21)
    df['trend_str'] = (ema9 - ema21) / ema21

    vol_med = df['volume'].rolling(20).mean()
    df['vol_surge'] = df['volume'] / vol_med

    df['btc_mom'] = df['btc_close'].pct_change(5)
    df['rel_str'] = df['mom_5'] - df['btc_mom']

    # 3. Limpeza
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 4. Target (Para análise futura, a IA RL não usa isso diretamente)
    future_return = df['close'].shift(-HORIZONTE_ALVO) / df['close'] - 1
    vol_check = df['close'].pct_change().rolling(20).std()
    valid_trade = vol_check < 0.02 
    
    conditions = [
        (future_return > ALVO_LUCRO) & valid_trade,
        (future_return < -ALVO_LUCRO) & valid_trade
    ]
    choices = [1, 2]
    df['target'] = np.select(conditions, choices, default=0)

    return df.iloc[:-HORIZONTE_ALVO]

def main():
    con = BinanceConnector()
    print("👑 Baixando BTC Mestre...")
    df_btc = buscar_historico(con, "BTCUSDT")
    if df_btc is None: return

    moedas = obter_top_50_moedas(con)
    dfs = []
    
    print(f"🚜 Minerando V11 Fusion (Com 'close' incluso)...")
    for i, par in enumerate(moedas):
        if par == "BTCUSDT": continue
        print(f"[{i+1}/{len(moedas)}] {par}...", end="\r")
        
        df_raw = buscar_historico(con, par)
        if df_raw is not None and len(df_raw) > 1000:
            try:
                df_proc = processar_fusao(df_raw, df_btc)
                
                # --- CORREÇÃO: ADICIONADO 'close' À LISTA ---
                cols = [
                    'close',  # <--- ESSENCIAL PARA O GÊNESIS
                    'mom_3', 'mom_5', 'mom_10', 'vol_ratio', 'pos_canal', 
                    'trend_str', 'vol_surge', 'btc_mom', 'rel_str',
                    'target', 'timestamp'
                ]
                
                if all(c in df_proc.columns for c in cols):
                    dfs.append(df_proc[cols])
            except: pass
            
    if dfs:
        print("\n🌪️ Unificando V11...")
        df_final = pd.concat(dfs, ignore_index=True)
        df_final = df_final.sort_values('timestamp').reset_index(drop=True)
        df_final.drop(columns=['timestamp'], inplace=True)
        
        df_final.to_csv("dataset_v11_fusion.csv", index=False)
        print(f"\n💾 DATASET CORRIGIDO! {len(df_final)} linhas.")
        print("👉 Agora o 'train_genesis.py' vai funcionar!")
    else:
        print("\n❌ Erro na geração.")

if __name__ == "__main__":
    main()