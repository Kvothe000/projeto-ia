# Binance/gerar_dataset_v5.py
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
from indicators import Calculadora
import time

# --- CONFIGURAÇÕES DE "LEBRE" (MODO SCALPER) ---
PAR = "WLDUSDT"      
TIMEFRAME = "15m"
QTD_CANDLES = 25000  # Aumentamos um pouco para ter mais base
ALVO_LUCRO = 0.006   # 0.6% (Meta rápida)
ALVO_STOP = 0.003    # 0.3% (Se der errado, sai rápido)
FUTURO_VISAO = 4     # Olha apenas 1 hora (4 candles) para frente. Scalping é rápido!

def buscar_historico_massivo(connector, par, timeframe, qtd_total):
    print(f"🚜 Iniciando mineração V5 na {par}...")
    todos_candles = []
    end_time = None
    
    while len(todos_candles) < qtd_total:
        try:
            klines = connector.client.futures_klines(
                symbol=par, interval=timeframe, limit=1500, endTime=end_time
            )
            if not klines: break
            
            temp_df = pd.DataFrame(klines, columns=['timestamp', 'o', 'h', 'l', 'c', 'v', 'ct', 'qav', 'nt', 'tbb', 'tbq', 'ig'])
            todos_candles.extend(klines)
            end_time = int(temp_df.iloc[0]['timestamp']) - 1
            print(f"📥 Baixados: {len(todos_candles)} / {qtd_total}...")
            time.sleep(0.2)
        except Exception as e:
            print(f"❌ Erro: {e}")
            break
            
    todos_candles.sort(key=lambda x: x[0])
    return connector._tratar_df(todos_candles)

def calcular_vwap_intraday(df):
    df['time_obj'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['pv'] = df['close'] * df['volume']
    df['vwap_m'] = df.groupby(df['time_obj'].dt.date)['pv'].cumsum()
    df['vol_m'] = df.groupby(df['time_obj'].dt.date)['volume'].cumsum()
    df['VWAP'] = df['vwap_m'] / df['vol_m']
    df.drop(columns=['time_obj', 'pv', 'vwap_m', 'vol_m'], inplace=True)
    return df

def criar_features(df):
    print("🧠 Calculando Indicadores...")
    df = Calculadora.adicionar_todos(df)
    df = calcular_vwap_intraday(df)
    
    df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
    df['CVD_Slope'] = df['CVD'].diff(3)
    df['Vol_Relativo'] = df['volume'] / df['volume'].rolling(20).mean()
    df['ATRr'] = df['ATRr_14']
    return df

def criar_alvo_bidirecional(df):
    """Gera alvo: 0=Nada, 1=LONG, 2=SHORT"""
    targets = []
    print("🔮 Criando Gabarito (Long & Short)...")
    total = len(df)
    
    # Cache para velocidade
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(total):
        if i + FUTURO_VISAO >= total:
            targets.append(0)
            continue
        
        entry = closes[i]
        
        # Alvos LONG
        tp_long = entry * (1 + ALVO_LUCRO)
        sl_long = entry * (1 - ALVO_STOP)
        
        # Alvos SHORT
        tp_short = entry * (1 - ALVO_LUCRO)
        sl_short = entry * (1 + ALVO_STOP)
        
        res = 0 # 0 = Neutro/Loss
        
        for j in range(1, FUTURO_VISAO + 1):
            curr_h = highs[i+j]
            curr_l = lows[i+j]
            
            # Checa LONG
            if curr_l <= sl_long: # Stopou Long
                long_valid = False
            elif curr_h >= tp_long: # Gain Long
                res = 1
                break
                
            # Checa SHORT
            if curr_h >= sl_short: # Stopou Short
                short_valid = False
            elif curr_l <= tp_short: # Gain Short
                res = 2
                break
        
        targets.append(res)
    return targets

def main():
    con = BinanceConnector()
    df = buscar_historico_massivo(con, PAR, TIMEFRAME, QTD_CANDLES)
    if df is None: return

    df = criar_features(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['target'] = criar_alvo_bidirecional(df)
    df = df.iloc[:-FUTURO_VISAO]
    
    cols = ['RSI_14', 'ADX_14', 'Dist_VWAP', 'CVD_Slope', 'Vol_Relativo', 'ATRr', 'target']
    final_df = df[[c for c in cols if c in df.columns]]
    
    final_df.to_csv("dataset_v5_wld.csv", index=False)
    
    longs = len(final_df[final_df['target']==1])
    shorts = len(final_df[final_df['target']==2])
    total = len(final_df)
    
    print(f"\n💾 ARQUIVO V5 GERADO: dataset_v5_wld.csv")
    print(f"📈 Oportunidades LONG: {longs} ({(longs/total)*100:.1f}%)")
    print(f"📉 Oportunidades SHORT: {shorts} ({(shorts/total)*100:.1f}%)")
    print(f"🏆 Total de Wins Possíveis: {(longs+shorts)/total*100:.1f}%")

if __name__ == "__main__":
    main()