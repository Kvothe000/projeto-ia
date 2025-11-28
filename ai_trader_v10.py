# Binance/ai_trader_v10.py
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import os

class TraderIAV10:
    def __init__(self):
        self.modelo = None
        self.limiar = 0.55
        if os.path.exists("modelo_ia_v10.pkl"):
            self.modelo = joblib.load("modelo_ia_v10.pkl")
            print("🧠 Cérebro V10 Carregado!")
        else:
            print("❌ ERRO: modelo_ia_v10.pkl não encontrado.")

    def preparar_dados(self, df):
        # Features Conservadoras V10
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df['ema_cross'] = df['EMA_9'] - df['EMA_21']
        
        df.ta.atr(length=14, append=True)
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        vol_sma = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / vol_sma
        
        df['resistance_20'] = df['high'].rolling(20).max()
        df['support_20'] = df['low'].rolling(20).min()
        df['dist_resistance'] = (df['close'] - df['resistance_20']) / df['resistance_20']
        df['dist_support'] = (df['close'] - df['support_20']) / df['support_20']
        
        df.fillna(0, inplace=True)
        return df.iloc[[-1]]

    def analisar_mercado(self, df):
        if self.modelo is None: return "NEUTRO", 0.0, 0.0
        try:
            X = self.preparar_dados(df.copy())
            cols = [
                'RSI_14', 'ema_cross', 'volatility_20', 'momentum_5', 'momentum_10',
                'volume_ratio', 'dist_resistance', 'dist_support', 'ATR_14'
            ]
            # Garante ordem das colunas
            for c in cols: 
                if c not in X.columns: X[c] = 0
            
            probs = self.modelo.predict_proba(X[cols])[0]
            # Classes: 0=Neutro, 1=Long, 2=Short
            prob_long, prob_short = probs[1], probs[2]
            
            sinal = "NEUTRO"
            conf = max(probs)
            
            if prob_long > self.limiar: sinal = "BUY"
            elif prob_short > self.limiar: sinal = "SELL"
            
            atr = float(X['ATR_14'].values[0])
            atr_pct = (atr / X['close'].values[0]) * 100
            
            return sinal, conf, atr_pct
        except Exception as e:
            print(f"Erro V10: {e}")
            return "NEUTRO", 0.0, 0.0