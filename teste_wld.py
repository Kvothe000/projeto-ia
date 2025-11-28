# teste_wld.py
import pandas as pd
from binance_connector import BinanceConnector
from indicators import Calculadora, Estrategia

print("--- DIAGNÓSTICO WLDBTC ---")
connector = BinanceConnector()
df = connector.buscar_candles("WLDBTC", "15m", lookback="10 days ago UTC")

if df is not None:
    df = Calculadora.adicionar_todos(df)
    c_atual = df.iloc[-1]
    
    print(f"Preço Atual: {c_atual['close']:.8f}")
    
    # 1. Checa Exaustão
    dist_media = (c_atual['close'] - c_atual['EMA_21']) / c_atual['EMA_21'] * 100
    print(f"Distância da Média: {dist_media:.2f}% (Limite é 5.0%)")
    if dist_media > 5.0: print("❌ REJEITADO POR EXAUSTÃO (Esticado)")
    
    # 2. Checa Regime
    regime = Estrategia.obter_regime_mercado(df.iloc[-2])
    print(f"Regime Detectado: {regime}")
    
    # 3. Checa Sinais
    msgs, _ = Estrategia.analisar_sinais(df)
    if msgs:
        print(f"✅ SINAL TÉCNICO ENCONTRADO: {msgs[0]}")
    else:
        print("❌ NENHUM SETUP TÉCNICO IDENTIFICADO (Não fez Pullback, Squeeze ou Reversão)")
else:
    print("Erro ao baixar dados.")