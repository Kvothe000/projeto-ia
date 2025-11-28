# Genesis_AI/features_engine.py
import pandas as pd
import pandas_ta as ta
import numpy as np
import sys
import os

# Tenta importar a Calculadora da pasta Binance se necessário, 
# mas vamos re-implementar o essencial aqui para ser autônomo.
class FeaturesEngine:
    """
    O Motor de Engenharia de Features do Gênesis.
    Garante consistência matemática absoluta entre Treino e Execução.
    """
    
    @staticmethod
    def processar_dados(df_moeda, df_btc):
        """
        Recebe: DataFrame da Moeda e DataFrame do Bitcoin (OHLCV).
        Retorna: DataFrame pronto com todas as features calculadas e normalizadas.
        """
        try:
            # 1. Sincronização (Merge com BTC)
            # Precisamos garantir que os timestamps batem
            df_moeda = df_moeda.copy()
            df_btc = df_btc.copy()
            
            df_moeda['timestamp'] = df_moeda['timestamp'].astype('int64')
            df_btc['timestamp'] = df_btc['timestamp'].astype('int64')
            
            df_moeda = df_moeda.set_index('timestamp')
            df_btc_indexed = df_btc.set_index('timestamp')[['close', 'volume']]
            df_btc_indexed.columns = ['btc_close', 'btc_volume']
            
            # Inner Join: Só mantém candles onde temos dados de AMBOS
            df = df_moeda.join(df_btc_indexed, how='inner').reset_index()
            
            # Garante tipos numéricos
            for c in ['close', 'high', 'low', 'volume', 'btc_close', 'btc_volume']:
                df[c] = df[c].astype(float)

            # -----------------------------------------------------------
            # 2. ENGENHARIA DE FEATURES (V11 FUSION)
            # -----------------------------------------------------------

            # A. Momentum Multi-Tempo
            for p in [3, 5, 10, 20]:
                df[f'mom_{p}'] = df['close'].pct_change(p)

            # B. Volatilidade Relativa
            vol_curta = df['close'].pct_change().rolling(5).std()
            vol_longa = df['close'].pct_change().rolling(20).std()
            df['vol_ratio'] = vol_curta / vol_longa

            # C. Posição no Canal (Donchian)
            max_20 = df['high'].rolling(20).max()
            min_20 = df['low'].rolling(20).min()
            # Evita divisão por zero se max == min
            denom = max_20 - min_20
            df['pos_canal'] = np.where(denom == 0, 0.5, (df['close'] - min_20) / denom)

            # D. Força da Tendência (EMA)
            ema9 = df.ta.ema(close=df['close'], length=9)
            ema21 = df.ta.ema(close=df['close'], length=21)
            df['trend_str'] = (ema9 - ema21) / ema21

            # E. Explosão de Volume (Volume Surge)
            vol_med = df['volume'].rolling(20).mean()
            df['vol_surge'] = df['volume'] / vol_med

            # F. Contexto Bitcoin (O Chefe)
            df['btc_mom'] = df['btc_close'].pct_change(5)
            df['rel_str'] = df['mom_5'] - df['btc_mom'] # Força Relativa

            # 3. Limpeza e Tratamento
            # Remove infinitos gerados por divisão por zero
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Preenche NaNs iniciais com 0 (para não perder dados no live)
            # No treino podemos dropar, mas no live precisamos da última linha
            df.fillna(0, inplace=True)

            return df

        except Exception as e:
            print(f"❌ Erro no Features Engine: {e}")
            return None

    @staticmethod
    def colunas_finais():
        """Retorna a lista exata de colunas que a IA espera"""
        return [
            'mom_3', 'mom_5', 'mom_10', 
            'vol_ratio', 'pos_canal', 
            'trend_str', 'vol_surge', 
            'btc_mom', 'rel_str'
        ]