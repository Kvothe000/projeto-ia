# Binance/ai_trader_v6.py (CÉREBRO COM CONTEXTO + ATR)
import joblib
import pandas as pd
import numpy as np
import os
from indicators import Calculadora
from binance_connector import BinanceConnector

class TraderIAV6:
    def __init__(self):
        self.modelo = None
        self.limiar = 0.55 # Confiança mínima (validada no treino)
        self.connector = BinanceConnector()
        self.carregar_modelo()

    def carregar_modelo(self):
        if os.path.exists("modelo_ia_v6.pkl"):
            self.modelo = joblib.load("modelo_ia_v6.pkl")
            print("🧠 Cérebro V6 (Context Aware) Carregado!")
        else:
            print("❌ ERRO: modelo_ia_v6.pkl não encontrado.")

    def preparar_dados(self, df_moeda):
        # 1. Baixar BTC para contexto (Últimos 100 candles para garantir médias)
        # O df_moeda tem 500 candles, baixamos 500 do BTC
        df_btc = self.connector.buscar_candles("BTCUSDT", "15m", limit=len(df_moeda))
        if df_btc is None: return None

        # 2. Sincronizar (Merge)
        df_moeda = df_moeda.set_index('timestamp')
        df_btc = df_btc.set_index('timestamp')[['close']]
        df_btc.columns = ['btc_close']
        
        # Junta os dados (Inner Join)
        df = df_moeda.join(df_btc, how='inner').reset_index()

        # 3. Calcular Indicadores (IGUAL AO TREINO)
        df = Calculadora.adicionar_todos(df)
        
        # VWAP
        df['pv'] = df['close'] * df['volume']
        df['VWAP'] = df['pv'].cumsum() / df['volume'].cumsum()
        df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
        
        # CVD Normalizado
        vol_media = df['volume'].rolling(20).mean()
        df['CVD_Slope'] = df['CVD'].diff(3) / vol_media
        df['Vol_Relativo'] = df['volume'] / vol_media
        df['ATRr'] = df['ATRr_14']
        
        # Contexto BTC
        df['BTC_Change'] = df['btc_close'].pct_change(3)
        df['Rel_Strength'] = df['close'].pct_change(3) - df['BTC_Change']
        
        df.replace([np.inf, -np.inf], 0, inplace=True)
        return df.iloc[[-1]].fillna(0) # Retorna só a última linha

    def analisar_mercado(self, df_candles):
        if self.modelo is None: return "NEUTRO", 0.0, 0.0

        try:
            X_hoje = self.preparar_dados(df_candles.copy())
            if X_hoje is None: return "NEUTRO", 0.0, 0.0
            
            # Colunas exatas do treino V6
            cols = ['RSI_14', 'ADX_14', 'Dist_VWAP', 'CVD_Slope', 'Vol_Relativo', 'ATRr', 'BTC_Change', 'Rel_Strength']
            
            # Previsão
            probs = self.modelo.predict_proba(X_hoje[cols])[0]
            prob_long, prob_short = probs[1], probs[2]
            
            sinal = "NEUTRO"
            confianca = max(probs)
            
            if prob_long > self.limiar: sinal = "BUY"
            elif prob_short > self.limiar: sinal = "SELL"
            
            # Retorna também o ATR (em %) para calcular o Stop Loss
            atr_atual_pct = float(X_hoje['ATRr'].values[0])
            
            return sinal, confianca, atr_atual_pct

        except Exception as e:
            # print(f"Erro IA: {e}")
            return "NEUTRO", 0.0, 0.0