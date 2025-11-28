# Binance/ai_trader_v7.py
import joblib
import pandas as pd
import numpy as np
import os
from indicators import Calculadora
from binance_connector import BinanceConnector

class TraderIAV7:
    def __init__(self):
        self.modelo = None
        self.limiar = 0.55
        self.connector = BinanceConnector()
        self.carregar_modelo()

    def carregar_modelo(self):
        if os.path.exists("modelo_ia_v7.pkl"):
            self.modelo = joblib.load("modelo_ia_v7.pkl")
            print("🧠 Cérebro V7 (Temporal) Carregado!")
        else:
            print("❌ ERRO: modelo_ia_v7.pkl não encontrado.")

    def preparar_dados(self, df_moeda):
        try:
            # 1. Baixar BTC
            df_btc = self.connector.buscar_candles("BTCUSDT", "15m", limit=len(df_moeda))
            if df_btc is None: return None

            # 2. Sincronizar
            df_moeda = df_moeda.reset_index(drop=True).set_index('timestamp')
            df_btc = df_btc[['timestamp', 'close']].reset_index(drop=True).set_index('timestamp')
            df_btc.columns = ['btc_close']
            df = df_moeda.join(df_btc, how='inner').reset_index()

            # 3. Features V7
            df = Calculadora.adicionar_todos(df)
            
            df['pv'] = df['close'] * df['volume']
            df['VWAP'] = df['pv'].cumsum() / df['volume'].cumsum()
            df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
            
            # Volatilidade e Volume
            vol_mean = df['volume'].rolling(20).mean()
            vol_std = df['volume'].rolling(20).std()
            
            df['Vol_ZScore'] = (df['volume'] - vol_mean) / vol_std # <--- NOVO
            df['CVD_Slope'] = df['CVD'].diff(3) / vol_mean
            df['Vol_Relativo'] = df['volume'] / vol_mean
            
            if 'ATRr_14' in df.columns:
                df['ATRr'] = df['ATRr_14']
            else:
                df['ATRr'] = (df['high'] - df['low']) / df['close'] * 100

            # Contexto BTC
            df['BTC_Change'] = df['btc_close'].pct_change(3)
            df['Rel_Strength'] = df['close'].pct_change(3) - df['BTC_Change']
            
            # Features Temporais <--- NOVO
            dates = pd.to_datetime(df['timestamp'], unit='ms')
            df['Hour_Sin'] = np.sin(2 * np.pi * dates.dt.hour / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * dates.dt.hour / 24)
            
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)
            
            return df.iloc[[-1]]

        except Exception as e:
            print(f"⚠️ Erro Preparação V7: {e}")
            return None

    def analisar_mercado(self, df_candles):
        if self.modelo is None: return "NEUTRO", 0.0, 0.0

        try:
            X_hoje = self.preparar_dados(df_candles.copy())
            if X_hoje is None: return "NEUTRO", 0.0, 0.0
            
            cols_ia = [
                'RSI_14', 'ADX_14', 'Dist_VWAP', 'CVD_Slope', 
                'Vol_ZScore', 'ATRr', 'BTC_Change', 'Rel_Strength',
                'Hour_Sin', 'Hour_Cos'
            ]
            
            for col in cols_ia:
                if col not in X_hoje.columns: X_hoje[col] = 0.0

            probs = self.modelo.predict_proba(X_hoje[cols_ia])[0]
            prob_long, prob_short = probs[1], probs[2]
            
            sinal = "NEUTRO"
            confianca = max(probs)
            
            if prob_long > self.limiar: sinal = "BUY"
            elif prob_short > self.limiar: sinal = "SELL"
            
            atr_atual_pct = float(X_hoje['ATRr'].values[0])
            
            return sinal, confianca, atr_atual_pct

        except Exception as e:
            print(f"❌ Erro IA V7: {e}")
            return "NEUTRO", 0.0, 0.0