# Binance/gerar_dataset_v3.py (CORRIGIDO)
import pandas as pd
from binance.client import Client
import config
from indicators import Calculadora
import numpy as np

# --- CONFIGURAÇÕES ---
PAR = "BTCUSDT"
TIMEFRAME = Client.KLINE_INTERVAL_15MINUTE
QTD_CANDLES = 15000 
ALVO_LUCRO = 0.008   # 0.8%
ALVO_STOP = 0.005    # 0.5%
FUTURO_VISAO = 12    

def baixar_historico():
    print(f"🚀 Baixando {QTD_CANDLES} candles para turbinar a IA...")
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    # Cálculo aproximado de dias
    days_ago = int(QTD_CANDLES * 15 / 60 / 24) + 5
    klines = client.get_historical_klines(PAR, TIMEFRAME, f"{days_ago} days ago UTC")
    
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tbba', 'tbqa', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    for c in ['open', 'high', 'low', 'close', 'volume']: 
        df[c] = pd.to_numeric(df[c])
    return df

def criar_features_avancadas(df):
    print("🧠 Calculando 'Contexto' (Deltas e Volume Relativo)...")
    
    # 1. Indicadores Básicos
    df = Calculadora.adicionar_todos(df)
    
    # --- CORREÇÃO DE BUGS DE NOME DE COLUNA ---
    # Encontra o nome real das colunas BBU (Upper) e BBL (Lower)
    # Procura colunas que começam com BBU_20 e BBL_20
    col_bbu = next((c for c in df.columns if c.startswith('BBU_20')), None)
    col_bbl = next((c for c in df.columns if c.startswith('BBL_20')), None)
    
    if col_bbu is None or col_bbl is None:
        print(f"⚠️ AVISO: Colunas de Bollinger não encontradas. Colunas disponíveis: {list(df.columns)}")
        # Fallback para evitar crash
        df['Dist_BBU'] = 0
    else:
        # 2. NOVAS FEATURES (O Segredo)
        # Distância da Banda: Está esticada?
        df['Dist_BBU'] = df[col_bbu] - df['close']
        # Renomeia para garantir que o script de treino encontre
        df.rename(columns={col_bbu: 'BBU_20_2.0', col_bbl: 'BBL_20_2.0'}, inplace=True)

    # RSI Slope: O RSI está subindo ou descendo?
    if 'RSI_14' in df.columns:
        df['RSI_Slope'] = df['RSI_14'].diff(2)
    else:
        df['RSI_Slope'] = 0
    
    # Volume Power: O volume atual é maior que a média?
    df['Vol_Relativo'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Tamanho do Corpo: Candles grandes indicam força
    df['Corpo_Candle'] = (df['close'] - df['open']).abs() / df['open']
    
    # Distância da média de 21
    if 'EMA_21' in df.columns:
        df['DIST_EMA21'] = df['close'] - df['EMA_21']
    else:
        df['DIST_EMA21'] = 0
    
    return df

def criar_alvo(df):
    targets = []
    print("🔮 Definindo Gabarito...")
    total = len(df)
    
    for i in range(total):
        if i + FUTURO_VISAO >= total:
            targets.append(0)
            continue
        
        entry = df.iloc[i]['close']
        tp = entry * (1 + ALVO_LUCRO)
        stop = entry * (1 - ALVO_STOP)
        
        res = 0
        for j in range(1, FUTURO_VISAO + 1):
            row = df.iloc[i+j]
            if row['low'] <= stop: break
            if row['high'] >= tp: 
                res = 1
                break
        targets.append(res)
    return targets

def main():
    df = baixar_historico()
    df = criar_features_avancadas(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['target'] = criar_alvo(df)
    df = df.iloc[:-FUTURO_VISAO]
    
    # Selecionamos apenas as colunas "turbinadas" para a IA
    features = [
        'RSI_14', 'RSI_Slope',       # Força + Direção
        'Vol_Relativo',              # Interesse do mercado
        'ADX_14', 'ATRr_14',         # Tendência e Volatilidade
        'DIST_EMA21',                # Esticamento
        'Corpo_Candle',              # Ação do Preço pura
        'target'
    ]
    
    # Filtra colunas que realmente existem para evitar erro
    cols_existentes = [c for c in features if c in df.columns]
    
    final_df = df[cols_existentes]
    final_df.to_csv("dataset_v3_turbo.csv", index=False)
    print(f"✅ Dataset V3 gerado com {len(final_df)} linhas.")
    
    wins = final_df[final_df['target'] == 1].shape[0]
    print(f"📊 Wins no histórico: {wins} ({(wins/len(final_df))*100:.1f}%)")

if __name__ == "__main__":
    main()