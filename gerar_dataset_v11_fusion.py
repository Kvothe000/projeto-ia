# Binance/gerar_dataset_v11_fusion.py (CORRIGIDO)
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
import pandas_ta as ta
import time

# --- CONFIGURAÇÃO V11 ---
QTD_MOEDAS = 50
TIMEFRAME = "15m"
QTD_POR_MOEDA = 6000   # 6000 candles = ~2 meses (Rápido e Recente)
HORIZONTE_ALVO = 8     # 2 horas
ALVO_LUCRO = 0.008     # 0.8%

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
    # 1. Merge Inicial (Moeda + BTC)
    df_moeda = df_moeda.set_index('timestamp')
    df_btc = df_btc.set_index('timestamp')[['close', 'volume']]
    df_btc.columns = ['btc_close', 'btc_volume']
    
    # Inner Join para alinhar tempos
    df = df_moeda.join(df_btc, how='inner').reset_index()
    
    # Garante numérico
    cols_num = ['close', 'high', 'low', 'volume', 'btc_close', 'btc_volume']
    for c in cols_num: df[c] = df[c].astype(float)

    # 2. Features Matemáticas (V11)
    
    # Momentum
    for p in [3, 5, 10, 20]:
        df[f'mom_{p}'] = df['close'].pct_change(p)

    # Volatilidade Relativa
    vol_curta = df['close'].pct_change().rolling(5).std()
    vol_longa = df['close'].pct_change().rolling(20).std()
    df['vol_ratio'] = vol_curta / vol_longa

    # Posição no Canal
    max_20 = df['high'].rolling(20).max()
    min_20 = df['low'].rolling(20).min()
    df['pos_canal'] = (df['close'] - min_20) / (max_20 - min_20)

    # Tendência EMA
    ema9 = df.ta.ema(close=df['close'], length=9)
    ema21 = df.ta.ema(close=df['close'], length=21)
    df['trend_str'] = (ema9 - ema21) / ema21

    # Volume Surge
    vol_med = df['volume'].rolling(20).mean()
    df['vol_surge'] = df['volume'] / vol_med

    # Contexto BTC
    df['btc_mom'] = df['btc_close'].pct_change(5)
    df['rel_str'] = df['mom_5'] - df['btc_mom']

    # 3. Limpeza de Nulos (Crucial antes do Target)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 4. Target Inteligente
    # Como resetamos o index acima, o shift vai funcionar perfeitamente
    future_return = df['close'].shift(-HORIZONTE_ALVO) / df['close'] - 1
    
    # Filtro de Volatilidade (Não operar no caos)
    # Usamos a vol_longa que já calculamos (recuperando pelo index)
    # Nota: Como dropamos linhas, precisamos recalcular ou garantir alinhamento
    # Simplificando: Recalculamos vol_longa limpa
    vol_check = df['close'].pct_change().rolling(20).std()
    
    valid_trade = vol_check < 0.02 # < 2% volatilidade
    
    conditions = [
        (future_return > ALVO_LUCRO) & valid_trade,
        (future_return < -ALVO_LUCRO) & valid_trade
    ]
    choices = [1, 2]
    df['target'] = np.select(conditions, choices, default=0)

    # Remove os últimos candles que não têm futuro (Target NaN ou falso)
    return df.iloc[:-HORIZONTE_ALVO]

def main():
    con = BinanceConnector()
    print("👑 Baixando BTC Mestre...")
    df_btc = buscar_historico(con, "BTCUSDT")
    if df_btc is None: return

    moedas = obter_top_50_moedas(con)
    dfs = []
    
    print(f"🚜 Minerando V11 Fusion...")
    for i, par in enumerate(moedas):
        if par == "BTCUSDT": continue
        print(f"[{i+1}/{len(moedas)}] {par}...", end="\r")
        
        df_raw = buscar_historico(con, par)
        if df_raw is not None and len(df_raw) > 1000:
            try:
                df_proc = processar_fusao(df_raw, df_btc)
                
                cols = [
                    'mom_3', 'mom_5', 'mom_10', 'vol_ratio', 'pos_canal', 
                    'trend_str', 'vol_surge', 'btc_mom', 'rel_str',
                    'target', 'timestamp'
                ]
                # Verifica se colunas existem
                if all(c in df_proc.columns for c in cols):
                    dfs.append(df_proc[cols])
            except Exception as e:
                # print(f"Erro em {par}: {e}")
                pass
            
    if dfs:
        print("\n🌪️ Unificando V11...")
        df_final = pd.concat(dfs, ignore_index=True)
        df_final = df_final.sort_values('timestamp').reset_index(drop=True)
        df_final.drop(columns=['timestamp'], inplace=True)
        
        df_final.to_csv("dataset_v11_fusion.csv", index=False)
        wins = len(df_final[df_final['target']!=0])
        print(f"\n💾 DATASET V11 GERADO! {len(df_final)} linhas.")
        print(f"📊 Oportunidades: {wins} ({(wins/len(df_final))*100:.1f}%)")
    else:
        print("\n❌ Erro na geração.")

if __name__ == "__main__":
    main()