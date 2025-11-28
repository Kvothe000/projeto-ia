# Binance/ai_trader_v6.py (VERSÃO ESTÁVEL)
import joblib
import pandas as pd
import numpy as np
import os
from indicators import Calculadora
from binance_connector import BinanceConnector

class TraderIAV6:
    def __init__(self):
        self.modelo = None
        self.limiar = 0.55
        self.connector = BinanceConnector()
        self.carregar_modelo()

    def carregar_modelo(self):
        if os.path.exists("modelo_ia_v6.pkl"):
            self.modelo = joblib.load("modelo_ia_v6.pkl")
            print("🧠 Cérebro V6 Carregado!")
        else:
            print("❌ ERRO: modelo_ia_v6.pkl não encontrado.")

    def preparar_dados(self, df_moeda, df_btc=None): # <--- Aceita BTC externo
        try:
            # 1. Se não recebeu BTC, tenta baixar (Fallback)
            if df_btc is None:
                df_btc = self.connector.buscar_candles("BTCUSDT", "15m", limit=len(df_moeda))
            
            if df_btc is None or len(df_btc) < 10:
                # Se falhar, usa a moeda sozinha (Modo Emergência) para não zerar ATR
                df = df_moeda.copy()
                df['btc_close'] = df['close'] 
            else:
                # Sincroniza dados (Merge Seguro)
                df_moeda = df_moeda.set_index('timestamp')
                df_btc_index = df_btc[['timestamp', 'close']].set_index('timestamp')
                df_btc_index.columns = ['btc_close']
                df = df_moeda.join(df_btc_index, how='inner').reset_index()

            # 2. Calcular Indicadores (Com Calculadora Blindada)
            df = Calculadora.adicionar_todos(df)
            
            # 3. Features
            df['pv'] = df['close'] * df['volume']
            df['VWAP'] = df['pv'].cumsum() / df['volume'].cumsum()
            df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
            
            vol_media = df['volume'].rolling(20).mean()
            df['CVD_Slope'] = df['CVD'].diff(3) / vol_media
            df['Vol_Relativo'] = df['volume'] / vol_media
            
            # ATR (Se não existir, cria fallback)
            if 'ATRr_14' in df.columns:
                df['ATRr'] = df['ATRr_14']
            else:
                # Calcula ATRr na mão se a lib falhar
                df['ATRr'] = (df['high'] - df['low']) / df['close'] * 100

            # Contexto
            if 'btc_close' in df.columns:
                df['BTC_Change'] = df['btc_close'].pct_change(3)
                df['Rel_Strength'] = df['close'].pct_change(3) - df['BTC_Change']
            
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)
            
            return df.iloc[[-1]]

        except Exception as e:
            print(f"⚠️ Erro Preparação: {e}")
            return None

    def analisar_mercado(self, df_candles, df_btc=None): # <--- Aceita BTC aqui também
        if self.modelo is None: return "NEUTRO", 0.0, 0.0

        try:
            X_hoje = self.preparar_dados(df_candles.copy(), df_btc)
            if X_hoje is None: return "NEUTRO", 0.0, 0.0
            
            cols_ia = [
                'RSI_14', 'ADX_14', 'Dist_VWAP', 'CVD_Slope', 
                'Vol_Relativo', 'ATRr', 'BTC_Change', 'Rel_Strength'
            ]
            
            for col in cols_ia:
                if col not in X_hoje.columns: X_hoje[col] = 0.0

            probs = self.modelo.predict_proba(X_hoje[cols_ia])[0]
            prob_long, prob_short = probs[1], probs[2]
            
            sinal = "NEUTRO"
            confianca = max(probs)
            
            if prob_long > self.limiar: sinal = "BUY"
            elif prob_short > self.limiar: sinal = "SELL"
            
            atr = float(X_hoje['ATRr'].values[0])
            
            return sinal, confianca, atr

        except Exception as e:
            print(f"❌ Erro IA: {e}")
            return "NEUTRO", 0.0, 0.0