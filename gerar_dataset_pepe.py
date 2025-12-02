# Binance/gerar_dataset_pepe.py
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
import time
import sys
import os

# Importação do Motor (Mesma pasta)
from features_engine import FeaturesEngine

# --- CONFIGURAÇÃO ---
PAR = "1000PEPEUSDT" # Atenção: Em Futuros é 1000PEPE
TIMEFRAME = "15m"

def main():
    con = BinanceConnector()
    print(f"👑 Baixando BTC Mestre (Contexto)...")
    df_btc = con.buscar_candles("BTCUSDT", TIMEFRAME, limit=1500)
    
    if df_btc is None:
        print("❌ Erro ao baixar BTC.")
        return

    print(f"🐸 Baixando {PAR} em profundidade...")
    klines = []
    end_time = None
    
    # Baixa 15.000 candles (Histórico Longo)
    for _ in range(10): 
        k = con.client.futures_klines(symbol=PAR, interval=TIMEFRAME, limit=1500, endTime=end_time)
        if not k: break
        klines.extend(k)
        end_time = int(k[0][0]) - 1
        print(f"   📦 Baixados: {len(klines)} candles...", end='\r')
        time.sleep(0.2)
    
    print("")
    
    if not klines:
        print(f"❌ Erro: Nenhum dado para {PAR}. Verifique se o símbolo está correto.")
        return

    klines.sort(key=lambda x: x[0])
    df_raw = con._tratar_df(klines)
    
    print(f"🧠 Processando Features para {PAR}...")
    
    try:
        df_final = FeaturesEngine.processar_dados(df_raw, df_btc)
        
        if df_final is not None and not df_final.empty:
            if 'target' not in df_final.columns: df_final['target'] = 0 

            df_final.to_csv("dataset_pepe_clean.csv", index=False)
            print(f"✅ DATASET PEPE GERADO: {len(df_final)} linhas.")
            print(f"📂 Salvo em: Binance/dataset_pepe_clean.csv")
        else:
            print("❌ Erro: DataFrame vazio.")
            
    except Exception as e:
        print(f"❌ Erro processamento: {e}")

if __name__ == "__main__":
    main()