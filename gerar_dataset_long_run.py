# Binance/gerar_dataset_long_run.py - MINERAÇÃO DE 1 ANO
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
import time
import sys
import os

# GPS de Pastas
current_dir = os.path.dirname(os.path.abspath(__file__))
genesis_dir = os.path.abspath(os.path.join(current_dir, '..', 'Genesis_AI'))
sys.path.append(genesis_dir)

try:
    from features_engine import FeaturesEngine
except ImportError:
    # Tenta local se o GPS falhar
    from features_engine import FeaturesEngine

PAR = "WLDUSDT"
TIMEFRAME = "15m"
# 1 Ano = 365 dias * 24h * 4 (15m) = 35.040 candles
QTD_BLOCOS = 25 # 25 * 1500 = 37.500 candles

def baixar_historico_longo(con, simbolo):
    print(f"🚜 Baixando 1 ANO de {simbolo}...")
    klines = []
    end_time = None
    
    for i in range(QTD_BLOCOS): 
        k = con.client.futures_klines(symbol=simbolo, interval=TIMEFRAME, limit=1500, endTime=end_time)
        if not k: break
        klines.extend(k)
        end_time = int(k[0][0]) - 1
        print(f"   📦 Bloco {i+1}/{QTD_BLOCOS} baixado...", end='\r')
        time.sleep(0.2)
    
    print(f"\n✅ Total {simbolo}: {len(klines)} velas.")
    klines.sort(key=lambda x: x[0])
    return con._tratar_df(klines)

def main():
    con = BinanceConnector()
    
    # 1. Baixa WLD (1 Ano)
    df_wld = baixar_historico_longo(con, PAR)
    
    # 2. Baixa BTC (1 Ano - Para Contexto)
    df_btc = baixar_historico_longo(con, "BTCUSDT")

    if df_wld is None or df_btc is None:
        print("❌ Erro no download."); return

    print(f"🧠 Processando Engenharia de Features (Pode demorar)...")
    
    try:
        # Processa tudo
        df_final = FeaturesEngine.processar_dados(df_wld, df_btc)
        
        if df_final is not None and not df_final.empty:
            if 'target' not in df_final.columns: df_final['target'] = 0 

            # Salva
            arquivo_saida = "dataset_wld_1ano.csv"
            df_final.to_csv(arquivo_saida, index=False)
            print(f"✅ DATASET ANUAL GERADO: {len(df_final)} linhas.")
            print(f"📂 Arquivo: Binance/{arquivo_saida}")
        else:
            print("❌ Erro: DataFrame vazio.")
            
    except Exception as e:
        print(f"❌ Erro processamento: {e}")

if __name__ == "__main__":
    main()