# Binance/indicators.py (VERSÃO ATR MANUAL)
import pandas as pd
import pandas_ta as ta
import numpy as np
import config

class Calculadora:
    @staticmethod
    def adicionar_todos(df):
        # 1. LIMPEZA TOTAL
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Se for muito pequeno, devolve vazio mas estruturado
        if len(df) < 20:
            for col in ['ATR_14', 'ADX_14', 'RSI_14', 'ATRr_14']: df[col] = 0.0
            return df

        try:
            # 2. GARANTIA DE SÉRIE NUMÉRICA
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            volume = df['volume'].astype(float)

            # 3. CÁLCULO MANUAL DO ATR (A SALVAÇÃO) 🛡️
            # True Range = Max(H-L, Abs(H-PrevClose), Abs(L-PrevClose))
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            # Média Móvel do True Range (ATR 14)
            df['ATR_14'] = tr.rolling(14).mean().fillna(0)
            
            # ATR Percentual
            df['ATRr_14'] = (df['ATR_14'] / close * 100).fillna(0)

            # 4. OUTROS INDICADORES (via Lib ou Manual)
            df.ta.adx(length=14, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            
            # Médias
            df['EMA_9'] = df.ta.ema(close=close, length=9)
            df['EMA_21'] = df.ta.ema(close=close, length=21)
            df['EMA_200'] = df.ta.ema(close=close, length=200).fillna(0)
            
            df['VOL_SMA_20'] = volume.rolling(20).mean()
            
            # Preenche qualquer buraco restante com 0
            df.fillna(0, inplace=True)

        except Exception as e:
            print(f"⚠️ Erro Calculadora: {e}")
            # Em último caso, garante ATR mínimo para não travar
            df['ATR_14'] = df['close'] * 0.01 
            df['ATRr_14'] = 1.0
            
        return df