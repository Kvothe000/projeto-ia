# Binance/gerar_dataset_v11_fusion.py (REFATORADO)
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Genesis_AI'))
from features_engine import FeaturesEngine

# --- CONFIGURAÇÃO V11 ---
QTD_MOEDAS = 50
TIMEFRAME = "15m"
QTD_POR_MOEDA = 6000
HORIZONTE_ALVO = 8
ALVO_LUCRO = 0.008


def obter_top_50_moedas(connector):
    """Obtém as top 50 moedas por volume"""
    try:
        tickers = connector.client.futures_ticker()
        df = pd.DataFrame(tickers)
        df = df[df['symbol'].str.endswith('USDT')]
        df['quoteVolume'] = pd.to_numeric(df['quoteVolume'])
        return df.sort_values('quoteVolume', ascending=False).head(QTD_MOEDAS)['symbol'].tolist()
    except Exception as e:
        print(f"❌ Erro ao obter top moedas: {e}")
        return []


def buscar_historico(connector, par):
    """Busca histórico de candles com paginação"""
    try:
        klines = []
        end_time = None
        
        for _ in range(5):
            k = connector.client.futures_klines(
                symbol=par,
                interval=TIMEFRAME,
                limit=1500,
                endTime=end_time
            )
            if not k:
                break
            klines.extend(k)
            end_time = int(k[0][0]) - 1
            time.sleep(0.1)
            
        klines.sort(key=lambda x: x[0])
        return connector._tratar_df(klines)
    except Exception as e:
        print(f"❌ Erro ao buscar histórico {par}: {e}")
        return None


def calcular_target(df, horizonte=HORIZONTE_ALVO, alvo_lucro=ALVO_LUCRO):
    """Calcula o target para treinamento supervisionado"""
    future_return = df['close'].shift(-horizonte) / df['close'] - 1
    vol_check = df['close'].pct_change().rolling(20).std()
    valid_trade = vol_check < 0.02
    
    conditions = [
        (future_return > alvo_lucro) & valid_trade,
        (future_return < -alvo_lucro) & valid_trade
    ]
    choices = [1, 2]
    
    return np.select(conditions, choices, default=0)


def processar_fusao(df_moeda, df_btc):
    """Processa e fusiona dados da moeda com BTC usando FeaturesEngine"""
    # Usa o motor central de features
    df = FeaturesEngine.processar_dados(df_moeda, df_btc)
    
    # Adiciona target apenas para treinamento
    df['target'] = calcular_target(df)
    
    # Remove últimas linhas para evitar data leakage
    return df.iloc[:-HORIZONTE_ALVO]


def main():
    """Função principal"""
    con = BinanceConnector()
    
    print("👑 Baixando BTC Mestre...")
    df_btc = buscar_historico(con, "BTCUSDT")
    if df_btc is None:
        print("❌ Falha ao carregar dados do BTC")
        return

    moedas = obter_top_50_moedas(con)
    if not moedas:
        print("❌ Nenhuma moeda encontrada")
        return

    dfs = []
    print(f"🚜 Minerando V11 Fusion ({len(moedas)} moedas)...")
    
    for i, par in enumerate(moedas):
        if par == "BTCUSDT":
            continue
            
        print(f"[{i+1}/{len(moedas)}] Processando {par}...", end="\r")
        
        df_raw = buscar_historico(con, par)
        if df_raw is not None and len(df_raw) > 1000:
            try:
                df_proc = processar_fusao(df_raw, df_btc)
                
                # Mantém todas as features geradas pelo FeaturesEngine
                # Inclui automaticamente 'close' e outras features
                if len(df_proc) > 0:
                    dfs.append(df_proc)
                    
            except Exception as e:
                print(f"❌ Erro ao processar {par}: {e}")
                continue

    if dfs:
        print("\n🌪️ Unificando dataset...")
        df_final = pd.concat(dfs, ignore_index=True)
        df_final = df_final.sort_values('timestamp').reset_index(drop=True)
        
        # Remove colunas temporárias se existirem
        cols_to_drop = ['timestamp']
        df_final = df_final.drop(columns=[col for col in cols_to_drop if col in df_final.columns])
        
        # Salva dataset
        filename = "dataset_v11_fusion.csv"
        df_final.to_csv(filename, index=False)
        
        print(f"\n💾 DATASET SALVO! {len(df_final)} linhas, {len(df_final.columns)} features")
        print("📊 Features incluídas:", list(df_final.columns))
        print(f"📁 Arquivo: {filename}")
    else:
        print("\n❌ Nenhum dado válido foi processado")


if __name__ == "__main__":
    main()