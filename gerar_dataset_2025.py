# Binance/gerar_dataset_2025.py
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
import time
import sys
import os

# GPS de Pastas
current_dir = os.path.dirname(os.path.abspath(__file__))
# Tenta encontrar features_engine na mesma pasta ou na Genesis
try:
    from features_engine import FeaturesEngine
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'Genesis_AI')))
    from features_engine import FeaturesEngine

PAR = "WLDUSDT"
TIMEFRAME = "15m"
# 1 Ano ~ 35.000 candles
QTD_BLOCOS = 24 # 24 * 1500 = 36.000 candles

def main():
    con = BinanceConnector()
    print(f"📅 BAIXANDO DADOS DE 2025 ({PAR} + BTC)...")

    # 1. Baixa WLD
    klines_wld = []
    end_time = None
    print(f"   🐺 Baixando WLD...")
    for i in range(QTD_BLOCOS):
        k = con.client.futures_klines(symbol=PAR, interval=TIMEFRAME, limit=1500, endTime=end_time)
        if not k: break
        klines_wld.extend(k)
        end_time = int(k[0][0]) - 1
        print(f"      Bloco {i+1}/{QTD_BLOCOS}...", end='\r')
        time.sleep(0.1)
    
    # 2. Baixa BTC
    klines_btc = []
    end_time = None
    print(f"\n   👑 Baixando BTC...")
    for i in range(QTD_BLOCOS):
        k = con.client.futures_klines(symbol="BTCUSDT", interval=TIMEFRAME, limit=1500, endTime=end_time)
        if not k: break
        klines_btc.extend(k)
        end_time = int(k[0][0]) - 1
        print(f"      Bloco {i+1}/{QTD_BLOCOS}...", end='\r')
        time.sleep(0.1)

    print("\n🧠 Processando Features...")
    df_wld = con._tratar_df(klines_wld)
    df_btc = con._tratar_df(klines_btc)
    
    # Feature Engineering
    df_final = FeaturesEngine.processar_dados(df_wld, df_btc)
    
    if df_final is not None and not df_final.empty:
        # Adiciona target dummy se não tiver (para compatibilidade)
        if 'target' not in df_final.columns: df_final['target'] = 0
        
        arquivo = "dataset_2025.csv"
        df_final.to_csv(arquivo, index=False)
        print(f"✅ DATASET 2025 PRONTO: {len(df_final)} linhas.")
        print(f"📂 Salvo em: Binance/{arquivo}")
    else:
        print("❌ Erro ao gerar dataset.")

if __name__ == "__main__":
    main()