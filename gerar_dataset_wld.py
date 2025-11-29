# Binance/gerar_dataset_wld.py - FOCO TOTAL (IMPORT BLINDADO)
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
import time
import sys
import os

# --- CORREÇÃO DE IMPORTAÇÃO (GPS DE PASTAS) ---
# 1. Descobre onde este arquivo está (pasta Binance)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Define onde está a pasta Genesis_AI (Vizinha)
genesis_dir = os.path.abspath(os.path.join(current_dir, '..', 'Genesis_AI'))
# 3. Adiciona ao caminho do sistema
if genesis_dir not in sys.path:
    sys.path.append(genesis_dir)

try:
    from features_engine import FeaturesEngine
    print("✅ FeaturesEngine carregado com sucesso!")
except ImportError as e:
    print(f"❌ Erro Crítico: Não foi possível importar 'features_engine'.")
    print(f"   O Python procurou em: {genesis_dir}")
    print(f"   Erro detalhado: {e}")
    sys.exit(1)
# ----------------------------------------------

PAR = "WLDUSDT"
TIMEFRAME = "15m"

def main():
    con = BinanceConnector()
    print(f"👑 Baixando BTC Mestre (Contexto)...")
    # Baixa um pouco mais de BTC para garantir cobertura no merge
    df_btc = con.buscar_candles("BTCUSDT", TIMEFRAME, limit=1500)
    
    if df_btc is None:
        print("❌ Erro ao baixar BTC. Verifique a conexão.")
        return

    # Loop para baixar histórico longo da WLD (ex: 15.000 candles)
    print(f"🚜 Baixando {PAR} em profundidade...")
    klines = []
    end_time = None
    
    # Baixa 10 blocos de 1500 candles
    for _ in range(10): 
        k = con.client.futures_klines(symbol=PAR, interval=TIMEFRAME, limit=1500, endTime=end_time)
        if not k: break
        klines.extend(k)
        # Atualiza o tempo para pegar o bloco anterior
        end_time = int(k[0][0]) - 1
        print(f"   📦 Baixados: {len(klines)} candles...", end='\r')
        time.sleep(0.2)
    
    print("") # Quebra linha
        
    if not klines:
        print(f"❌ Nenhum dado baixado para {PAR}.")
        return

    # Ordena e converte
    klines.sort(key=lambda x: x[0])
    df_wld_raw = con._tratar_df(klines)
    
    print(f"🧠 Processando Engenharia de Features ({len(df_wld_raw)} linhas)...")
    
    # Usa o motor central do Gênesis para garantir compatibilidade
    try:
        # O FeaturesEngine faz o merge (Inner Join) e calcula tudo
        df_final = FeaturesEngine.processar_dados(df_wld_raw, df_btc)
        
        if df_final is not None and not df_final.empty:
            # Adiciona a coluna 'target' fictícia para o treino funcionar
            # (O ambiente vai ignorar, mas o script de treino pode exigir)
            if 'target' not in df_final.columns:
                 df_final['target'] = 0 

            # Salva com o nome correto para o treino WLD
            df_final.to_csv("dataset_wld_clean.csv", index=False)
            print(f"✅ DATASET WLD GERADO: {len(df_final)} linhas prontas.")
            print(f"📂 Arquivo salvo: Binance/dataset_wld_clean.csv")
            print("👉 Próximo passo: Rodar 'python Genesis_AI/train_genesis_wld.py'")
        else:
            print("❌ Erro: DataFrame final vazio após processamento (Verifique se as datas do BTC coincidem).")
            
    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()