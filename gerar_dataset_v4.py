# Binance/gerar_dataset_v4.py (VERSÃO MASSIVA)
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
from indicators import Calculadora
import time
from datetime import datetime

# --- CONFIGURAÇÕES ---
PAR = "BTCUSDT"
TIMEFRAME = "15m"
QTD_CANDLES = 30000  # Aprox 10 meses de dados (Agora vai!)
ALVO_LUCRO = 0.012   # 1.2%
ALVO_STOP = 0.006    # 0.6%
FUTURO_VISAO = 12    # 3 horas

def buscar_historico_massivo(connector, par, timeframe, qtd_total):
    """
    Faz um loop voltando no tempo para baixar MILHARES de candles de Futuros.
    A API limita 1500 por vez, então precisamos paginar.
    """
    print(f"🚜 Iniciando mineração profunda: Meta {qtd_total} candles...")
    
    todos_candles = []
    end_time = None # Começa do "agora"
    
    while len(todos_candles) < qtd_total:
        try:
            # Baixa o bloco (limitado a 1500 pela Binance)
            # Precisamos usar o cliente direto para ter controle do 'endTime'
            klines = connector.client.futures_klines(
                symbol=par, 
                interval=timeframe, 
                limit=1500,
                endTime=end_time
            )
            
            if not klines:
                print("⚠️ Sem mais dados disponíveis.")
                break
                
            # Tratamento básico (igual ao connector)
            # Colunas: Open time, Open, High, Low, Close, Volume, Close time...
            # Precisamos converter para DataFrame para pegar o timestamp do mais antigo
            temp_df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'num_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Adiciona na lista principal
            todos_candles.extend(klines)
            
            # Atualiza o end_time para pegar os candles ANTERIORES a esse bloco
            # O timestamp[0] é o mais antigo deste bloco.
            primeiro_timestamp = int(temp_df.iloc[0]['timestamp'])
            end_time = primeiro_timestamp - 1
            
            print(f"📥 Baixados: {len(todos_candles)} / {qtd_total}...")
            
            time.sleep(0.5) # Respeita a API
            
        except Exception as e:
            print(f"❌ Erro no download: {e}")
            break
            
    # Inverte a lista (porque baixamos do presente pro passado)
    todos_candles.sort(key=lambda x: x[0])
    
    # Processa via Connector para ter CVD e colunas certas
    return connector._tratar_df(todos_candles)

def calcular_vwap_intraday(df):
    df['time_obj'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['pv'] = df['close'] * df['volume']
    df['vwap_m'] = df.groupby(df['time_obj'].dt.date)['pv'].cumsum()
    df['vol_m'] = df.groupby(df['time_obj'].dt.date)['volume'].cumsum()
    df['VWAP'] = df['vwap_m'] / df['vol_m']
    df.drop(columns=['time_obj', 'pv', 'vwap_m', 'vol_m'], inplace=True)
    return df

def criar_features_institucionais(df):
    print("🧠 Calculando Indicadores Institucionais...")
    df = Calculadora.adicionar_todos(df)
    df = calcular_vwap_intraday(df)
    
    # Features Inteligentes
    df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
    df['CVD_Slope'] = df['CVD'].diff(3)
    df['Vol_Relativo'] = df['volume'] / df['volume'].rolling(20).mean()
    df['ATRr'] = df['ATRr_14']
    
    return df

def criar_alvo(df):
    targets = []
    print("🔮 Criando Gabarito (Isso pode demorar um pouco)...")
    total = len(df)
    precos = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Otimização com Numpy para ser rápido
    for i in range(total):
        if i + FUTURO_VISAO >= total:
            targets.append(0)
            continue
        
        entry = precos[i]
        tp = entry * (1 + ALVO_LUCRO)
        stop = entry * (1 - ALVO_STOP)
        
        res = 0
        # Olha o futuro
        for j in range(1, FUTURO_VISAO + 1):
            if lows[i+j] <= stop:
                res = 0
                break
            if highs[i+j] >= tp:
                res = 1
                break
        targets.append(res)
    return targets

def main():
    con = BinanceConnector()
    
    # 1. Download Massivo
    df = buscar_historico_massivo(con, PAR, TIMEFRAME, QTD_CANDLES)
    
    if df is None or len(df) < 1000:
        print("❌ Falha crítica: Poucos dados baixados.")
        return

    # 2. Engenharia de Features
    df = criar_features_institucionais(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 3. Alvo
    df['target'] = criar_alvo(df)
    df = df.iloc[:-FUTURO_VISAO]
    
    # 4. Salvar
    features = [
        'RSI_14', 'ADX_14', 
        'Dist_VWAP', 'CVD_Slope', 
        'Vol_Relativo', 'ATRr', 
        'target'
    ]
    
    cols_finais = [c for c in features if c in df.columns]
    
    nome_arquivo = "dataset_v4_institucional.csv"
    df[cols_finais].to_csv(nome_arquivo, index=False)
    
    wins = df[df['target'] == 1].shape[0]
    print(f"\n💾 ARQUIVO GERADO: {nome_arquivo}")
    print(f"📊 Total de Exemplos: {len(df)}")
    print(f"🏆 Oportunidades Reais de Lucro: {wins} ({(wins/len(df))*100:.2f}%)")
    print("👉 Agora rode o 'treinar_ia_v4.py' novamente!")

if __name__ == "__main__":
    main()