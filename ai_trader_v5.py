# Binance/ai_trader_v5.py
import joblib
import pandas as pd
import numpy as np
import os
from indicators import Calculadora

class TraderIAV5:
    def __init__(self):
        self.modelo = None
        # Defina aqui a certeza mínima que você testou e deu certo
        self.limiar_long = 0.60  # 60% para Comprar
        self.limiar_short = 0.60 # 60% para Vender
        self.carregar_modelo()

    def carregar_modelo(self):
        if os.path.exists("modelo_ia_v5.pkl"):
            self.modelo = joblib.load("modelo_ia_v5.pkl")
            print("🧠 Cérebro IA V5 (Long/Short) carregado com sucesso!")
        else:
            print("❌ ERRO CRÍTICO: modelo_ia_v5.pkl não encontrado.")

    def calcular_vwap_intraday(self, df):
        # A mesma lógica do gerador de dataset
        df['time_obj'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['pv'] = df['close'] * df['volume']
        df['vwap_m'] = df.groupby(df['time_obj'].dt.date)['pv'].cumsum()
        df['vol_m'] = df.groupby(df['time_obj'].dt.date)['volume'].cumsum()
        df['VWAP'] = df['vwap_m'] / df['vol_m']
        df.drop(columns=['time_obj', 'pv', 'vwap_m', 'vol_m'], inplace=True)
        return df

    def preparar_dados(self, df):
        # 1. Indicadores Básicos
        df = Calculadora.adicionar_todos(df)
        
        # 2. Indicadores Institucionais (IGUAL AO TREINO)
        df = self.calcular_vwap_intraday(df)
        
        df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
        # Se não tiver dados suficientes para diff(3), preenche com 0
        df['CVD_Slope'] = df['CVD'].diff(3).fillna(0)
        df['Vol_Relativo'] = df['volume'] / df['volume'].rolling(20).mean()
        df['ATRr'] = df['ATRr_14']
        
        # Seleciona apenas a última linha (o momento atual)
        # e as colunas exatas que a IA aprendeu
        cols_ia = ['RSI_14', 'ADX_14', 'Dist_VWAP', 'CVD_Slope', 'Vol_Relativo', 'ATRr']
        
        # Garante que não tem NaNs na última linha
        ultimo_candle = df.iloc[[-1]][cols_ia].fillna(0)
        return ultimo_candle

    def analisar_mercado(self, df_candles):
        if self.modelo is None: return "NEUTRO", 0.0

        try:
            X_hoje = self.preparar_dados(df_candles.copy())
            
            # Pede as probabilidades: [Neutro%, Long%, Short%]
            probs = self.modelo.predict_proba(X_hoje)[0]
            prob_neutro, prob_long, prob_short = probs[0], probs[1], probs[2]
            
            # Decisão baseada nos Limiares
            if prob_long >= self.limiar_long:
                return "BUY", prob_long
            elif prob_short >= self.limiar_short:
                return "SELL", prob_short
            else:
                # Retorna a maior probabilidade entre as neutras só para log
                return "NEUTRO", max(prob_neutro, prob_long, prob_short)
                
        except Exception as e:
            print(f"❌ Erro na IA: {e}")
            return "ERRO", 0.0