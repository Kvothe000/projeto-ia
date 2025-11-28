# Binance/gerar_dataset_v8.py (CHECK-UP DE DADOS)
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
from indicators import Calculadora
import time

QTD_MOEDAS = 50        
TIMEFRAME = "15m"
QTD_POR_MOEDA = 12000  
MULT_ATR_ALVO = 1.5   
MULT_ATR_STOP = 1.0   
FUTURO_VISAO = 12 

def obter_top_50_moedas(connector):
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
            k = connector.client.futures_klines(symbol=par, interval=TIMEFRAME, limit=1500, endTime=end_time)
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
    df = Calculadora.adicionar_todos(df) # Agora usa ATR Manual Seguro
    
    # Features
    df['pv'] = df['close'] * df['volume']
    df['VWAP'] = df['pv'].cumsum() / df['volume'].cumsum()
    df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
    
    vol_mean = df['volume'].rolling(20).mean()
    df['Vol_Relativo'] = df['volume'] / vol_mean
    df['CVD_Slope'] = df['CVD'].diff(3) / vol_mean
    vol_std = df['volume'].rolling(20).std()
    df['Vol_ZScore'] = (df['volume'] - vol_mean) / vol_std
    
    df['BTC_Change'] = df['btc_close'].pct_change(3)
    df['Rel_Strength'] = df['close'].pct_change(3) - df['BTC_Change']
    
    # Features Temporais
    dates = pd.to_datetime(df['timestamp'], unit='ms')
    df['Hour_Sin'] = np.sin(2 * np.pi * dates.dt.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * dates.dt.hour / 24)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    # Alvos Adaptativos
    targets = []
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['ATR_14'].values 
    
    total = len(df)
    
    for i in range(total):
        if i + FUTURO_VISAO >= total:
            targets.append(0); continue
            
        entry = closes[i]
        # Se ATR for zero (impossível agora), usa 0.5% do preço
        vol = atrs[i] if atrs[i] > 0 else (entry * 0.005)

        dist_win = vol * MULT_ATR_ALVO
        dist_loss = vol * MULT_ATR_STOP
        
        tp_long = entry + dist_win
        sl_long = entry - dist_loss
        tp_short = entry - dist_win
        sl_short = entry + dist_loss
        
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
    
    print(f"🚜 Minerando V8 (ATR Manual)...")
    
    for i, par in enumerate(moedas):
        if par == "BTCUSDT": continue
        print(f"[{i+1}/{len(moedas)}] {par}...", end="\r")
        
        df_raw = buscar_historico_rapido(con, par)
        if df_raw is not None and len(df_raw) > 1000:
            df_merged = adicionar_contexto_btc(df_raw, df_btc)
            df_proc = processar_dados(df_merged)
            
            # Validação rápida de Wins
            wins = len(df_proc[df_proc['target']!=0])
            if wins > 0:
                cols = [
                    'RSI_14', 'ADX_14', 'Dist_VWAP', 'CVD_Slope', 
                    'Vol_ZScore', 'ATRr_14', 'BTC_Change', 'Rel_Strength',
                    'Hour_Sin', 'Hour_Cos', 'target', 'timestamp'
                ]
                dfs.append(df_proc[cols])
            
    if dfs:
        print("\n🌪️ Consolidando Time Series V8...")
        df_final = pd.concat(dfs, ignore_index=True)
        df_final = df_final.sort_values('timestamp').reset_index(drop=True)
        df_final.drop(columns=['timestamp'], inplace=True)
        
        total = len(df_final)
        wins_total = len(df_final[df_final['target']!=0])
        
        if wins_total == 0:
            print("\n❌ ERRO CRÍTICO: 0 Oportunidades encontradas!")
            print("   Verifique se o timeframe 15m e o multiplicador ATR 1.5x fazem sentido.")
        else:
            df_final.to_csv("dataset_v8_atr.csv", index=False)
            print(f"\n💾 SUCESSO! dataset_v8_atr.csv gerado.")
            print(f"📊 Dados: {total} | Oportunidades: {wins_total} ({(wins_total/total)*100:.2f}%)")
    else:
        print("\n❌ Nenhum dado válido coletado.")

if __name__ == "__main__":
    main()