# Binance/features_engine.py (CÓPIA LOCAL PARA FACILITAR IMPORTAÇÃO)
import pandas as pd
import pandas_ta as ta
import numpy as np

class FeaturesEngine:
    """
    O Motor de Engenharia de Features (Versão Local Binance).
    """
    
    @staticmethod
    def processar_dados(df_moeda, df_btc):
        try:
            # 1. Preparação e Sincronização
            df = df_moeda.copy()
            btc = df_btc.copy()
            
            # Garante timestamp como inteiro
            df['timestamp'] = df['timestamp'].astype('int64')
            btc['timestamp'] = btc['timestamp'].astype('int64')
            
            # Indexa por tempo
            df = df.set_index('timestamp')
            btc_indexed = btc.set_index('timestamp')[['close', 'volume']]
            btc_indexed.columns = ['btc_close', 'btc_volume']
            
            # Inner Join: Sincroniza os candles
            df = df.join(btc_indexed, how='inner').reset_index()
            
            # Garante tipos numéricos
            cols_num = ['close', 'high', 'low', 'volume', 'btc_close', 'btc_volume']
            for c in cols_num:
                if c in df.columns:
                    df[c] = df[c].astype(float)

            # 2. Engenharia de Features (V11 FUSION)
            
            # Momentum
            for p in [3, 5, 10, 20]:
                df[f'mom_{p}'] = df['close'].pct_change(p)

            # Volatilidade Relativa
            vol_curta = df['close'].pct_change().rolling(5).std()
            vol_longa = df['close'].pct_change().rolling(20).std()
            df['vol_ratio'] = vol_curta / vol_longa

            # Posição no Canal
            max_20 = df['high'].rolling(20).max()
            min_20 = df['low'].rolling(20).min()
            denom = max_20 - min_20
            df['pos_canal'] = np.where(denom == 0, 0.5, (df['close'] - min_20) / denom)

            # Tendência
            ema9 = df.ta.ema(close=df['close'], length=9)
            ema21 = df.ta.ema(close=df['close'], length=21)
            df['trend_str'] = (ema9 - ema21) / ema21

            # Volume
            vol_med = df['volume'].rolling(20).mean()
            df['vol_surge'] = df['volume'] / vol_med

            # Contexto Bitcoin
            df['btc_mom'] = df['btc_close'].pct_change(5)
            df['rel_str'] = df['mom_5'] - df['btc_mom']

            # 3. Limpeza
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            return df

        except Exception as e:
            print(f"❌ Erro no Features Engine: {e}")
            return None

    @staticmethod
    def colunas_finais():
        return [
            'mom_3', 'mom_5', 'mom_10', 
            'vol_ratio', 'pos_canal', 
            'trend_str', 'vol_surge', 
            'btc_mom', 'rel_str'
        ]