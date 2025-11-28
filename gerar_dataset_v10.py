# Binance/gerar_dataset_v10.py - BASE LIMPA E BLINDADA
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
import pandas_ta as ta
import time

QTD_MOEDAS = 50        
TIMEFRAME = "15m"
QTD_POR_MOEDA = 5000   

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
        for _ in range(4): # 4 blocos = 6000 candles
            k = connector.client.futures_klines(symbol=par, interval=TIMEFRAME, limit=1500, endTime=end_time)
            if not k: break
            klines.extend(k)
            end_time = int(k[0][0]) - 1
            time.sleep(0.1)
        klines.sort(key=lambda x: x[0])
        return connector._tratar_df(klines)
    except: return None

def criar_features_conservadoras(df):
    """Features sugeridas pelo colega (Sem vazamento)"""
    # Garantia de tipos numéricos
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # 1. Indicadores Básicos
    df.ta.rsi(length=14, append=True)
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df['ema_cross'] = df['EMA_9'] - df['EMA_21']
    
    # 2. Volatilidade e ATR (COM PROTEÇÃO)
    df.ta.atr(length=14, append=True)
    
    # --- FIX DE SEGURANÇA: Se ATR_14 não for criado, calcula na mão ---
    if 'ATR_14' not in df.columns:
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()
    # -----------------------------------------------------------------
    
    # Preenche qualquer buraco com 0 para não crashar
    df['ATR_14'] = df['ATR_14'].fillna(0)

    # Volatilidade de 20 períodos
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    
    # 3. Momentum Simples (Variação passada)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    
    # 4. Volume Relativo
    vol_sma = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / vol_sma
    
    # 5. Distância de Suporte/Resistência (Donchian simples)
    df['resistance_20'] = df['high'].rolling(20).max()
    df['support_20'] = df['low'].rolling(20).min()
    
    # Normalização Simples
    df['dist_resistance'] = (df['close'] - df['resistance_20']) / df['resistance_20']
    df['dist_support'] = (df['close'] - df['support_20']) / df['support_20']

    # TARGET: Retorno futuro de 5 candles
    future_return = df['close'].shift(-5) / df['close'] - 1
    
    conditions = [
        future_return > 0.005,  # +0.5%
        future_return < -0.005  # -0.5%
    ]
    choices = [1, 2] # 1=Long, 2=Short
    df['target'] = np.select(conditions, choices, default=0) 
    
    # Limpeza Final
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True) # Garante que não sobra nenhum NaN
    
    return df

def main():
    con = BinanceConnector()
    moedas = obter_top_50_moedas(con)
    dfs = []
    
    print(f"🚜 Minerando V10 (Protocolo Seguro)...")
    for i, par in enumerate(moedas):
        if par == "BTCUSDT": continue
        print(f"[{i+1}/{len(moedas)}] {par}...", end="\r")
        
        df_raw = buscar_historico(con, par)
        if df_raw is not None and len(df_raw) > 1000:
            df_proc = criar_features_conservadoras(df_raw)
            
            cols = [
                'RSI_14', 'ema_cross', 'volatility_20', 'momentum_5', 'momentum_10',
                'volume_ratio', 'dist_resistance', 'dist_support', 'ATR_14',
                'target', 'timestamp'
            ]
            
            # Verifica se todas as colunas existem antes de adicionar
            cols_existentes = [c for c in cols if c in df_proc.columns]
            if len(cols_existentes) == len(cols):
                dfs.append(df_proc[cols])
            
    if dfs:
        print("\n🌪️ Consolidando V10...")
        df_final = pd.concat(dfs, ignore_index=True)
        df_final = df_final.sort_values('timestamp').reset_index(drop=True)
        df_final.drop(columns=['timestamp'], inplace=True)
        
        df_final.to_csv("dataset_v10_seguro.csv", index=False)
        print(f"\n💾 SUCESSO! dataset_v10_seguro.csv gerado.")
        
        counts = df_final['target'].value_counts()
        print(f"📊 Distribuição: {counts.to_dict()} (0=Neutro, 1=Long, 2=Short)")
    else:
        print("\n❌ Nenhum dado válido gerado.")

if __name__ == "__main__":
    main()