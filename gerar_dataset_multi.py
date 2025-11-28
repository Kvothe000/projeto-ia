# Binance/gerar_dataset_multi.py (MINERADOR UNIVERSAL)
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
from indicators import Calculadora
import time

# --- CONFIGURAÇÃO DE "BIG DATA" ---
MOEDAS_TREINO = [
    'BTCUSDT', 'ETHUSDT',       # Os Reis (Estabilidade)
    'SOLUSDT', 'BNBUSDT',       # Os Príncipes (Tendência)
    'DOGEUSDT', '1000PEPEUSDT', # As Memes (Caos/Volatilidade)
    'WLDUSDT', 'FETUSDT',       # As Tech/IA (Hype)
    'TRADOORUSDT', 'TURBOUSDT'  # As Novatas (Explosão)
]

TIMEFRAME = "15m"
QTD_POR_MOEDA = 15000  # ~5 meses de cada moeda
# Total = 150.000 candles de treino (Isso é nível profissional!)

# Alvos (Ajustados para capturar movimentos médios)
ALVO_LUCRO = 0.008   # 0.8% (Meta factível para todas)
ALVO_STOP = 0.004    # 0.4% (Stop curto)
FUTURO_VISAO = 4     # Olha 1 hora para frente

def buscar_historico_massivo(connector, par, timeframe, qtd_total):
    print(f"🚜 Minerando {par} ({qtd_total} candles)...")
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
            # print(f"   ↳ Progresso: {len(todos_candles)}...")
            time.sleep(0.1) # Respeita a API
        except Exception as e:
            print(f"❌ Erro em {par}: {e}")
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

def processar_moeda(df):
    # Engenharia de Features
    df = Calculadora.adicionar_todos(df)
    df = calcular_vwap_intraday(df)
    
    df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
    df['CVD_Slope'] = df['CVD'].diff(3).fillna(0)
    df['Vol_Relativo'] = df['volume'] / df['volume'].rolling(20).mean()
    df['ATRr'] = df['ATRr_14']
    
    # Criação do Alvo (Gabarito)
    targets = []
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    total = len(df)
    
    for i in range(total):
        if i + FUTURO_VISAO >= total:
            targets.append(0)
            continue
        
        entry = closes[i]
        tp_long = entry * (1 + ALVO_LUCRO)
        sl_long = entry * (1 - ALVO_STOP)
        tp_short = entry * (1 - ALVO_LUCRO)
        sl_short = entry * (1 + ALVO_STOP)
        
        res = 0
        for j in range(1, FUTURO_VISAO + 1):
            # Check Long
            if lows[i+j] <= sl_long: long_dead = True
            elif highs[i+j] >= tp_long: 
                res = 1; break # WIN LONG
            
            # Check Short (Se não deu Long, vê se dava Short)
            if highs[i+j] >= sl_short: short_dead = True
            elif lows[i+j] <= tp_short: 
                res = 2; break # WIN SHORT
        
        targets.append(res)
    
    df['target'] = targets
    return df.iloc[:-FUTURO_VISAO] # Remove o final sem futuro

def main():
    con = BinanceConnector()
    dfs_finais = []
    
    print(f"🌍 Iniciando GERAÇÃO DE DATASET UNIVERSAL...")
    print(f"🎯 Alvo: 10 Moedas x {QTD_POR_MOEDA} Candles = ~150.000 Exemplos\n")

    for moeda in MOEDAS_TREINO:
        try:
            df_raw = buscar_historico_massivo(con, moeda, TIMEFRAME, QTD_POR_MOEDA)
            if df_raw is not None and len(df_raw) > 1000:
                df_proc = processar_moeda(df_raw)
                
                # Seleciona colunas para IA
                cols = ['RSI_14', 'ADX_14', 'Dist_VWAP', 'CVD_Slope', 'Vol_Relativo', 'ATRr', 'target']
                df_clean = df_proc[[c for c in cols if c in df_proc.columns]].copy()
                
                dfs_finais.append(df_clean)
                wins = len(df_clean[df_clean['target']!=0])
                print(f"✅ {moeda}: Processado com sucesso ({len(df_clean)} linhas, {wins} wins)")
            else:
                print(f"⚠️ {moeda}: Dados insuficientes.")
        except Exception as e:
            print(f"❌ Falha crítica em {moeda}: {e}")

    # Junta tudo num arquivo só
    if dfs_finais:
        df_master = pd.concat(dfs_finais, ignore_index=True)
        # Embaralha os dados para a IA não viciar na ordem das moedas
        df_master = df_master.sample(frac=1).reset_index(drop=True)
        
        df_master.to_csv("dataset_universe.csv", index=False)
        print(f"\n💾 ARQUIVO FINAL GERADO: dataset_universe.csv")
        print(f"📊 Total de Exemplos para Treino: {len(df_master)}")
        print("👉 Agora rode o 'treinar_ia_multi.py'!")
    else:
        print("❌ Nenhum dado foi gerado.")

if __name__ == "__main__":
    main()