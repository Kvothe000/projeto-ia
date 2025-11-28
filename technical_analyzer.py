# Binance/technical_analyzer.py
import pandas as pd
import pandas_ta as ta

class AnalisadorTecnico:
    def adicionar_indicadores_completos(self, df):
        """Adiciona indicadores técnicos essenciais para análise"""
        try:
            # Garante que temos dados suficientes
            if len(df) < 50: return df

            # Indicadores de Tendência
            df.ta.ema(length=9, append=True)
            df.ta.ema(length=21, append=True)
            df.ta.ema(length=200, append=True)
            df.ta.adx(length=14, append=True)

            # Indicadores de Momentum
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            
            # Volatilidade
            df.ta.atr(length=14, append=True)
            df.ta.bbands(length=20, std=2, append=True)

            return df
        except Exception as e:
            print(f"❌ Erro indicadores técnicos: {e}")
            return df

    def calcular_momentum_score(self, df):
        """Calcula score de 0 a 1 indicando força da tendência"""
        try:
            score = 0
            max_score = 4
            candle = df.iloc[-1]
            
            # 1. ADX Forte
            if candle.get('ADX_14', 0) > 25: score += 1
            
            # 2. RSI Saudável (Não sobrecomprado demais em tendência)
            rsi = candle.get('RSI_14', 50)
            if 40 < rsi < 70: score += 1
            
            # 3. MACD Cruzado para cima
            if candle.get('MACD_12_26_9', 0) > candle.get('MACDs_12_26_9', 0): score += 1
            
            # 4. Volume acima da média
            vol_media = df['volume'].rolling(20).mean().iloc[-1]
            if vol_media > 0 and candle['volume'] > vol_media: score += 1
            
            return score / max_score
        except: return 0

    def detectar_padroes_velas(self, df):
        """Identifica padrões de reversão ou continuação"""
        padroes = []
        try:
            atual = df.iloc[-1]
            anterior = df.iloc[-2]
            
            corpo = abs(atual['close'] - atual['open'])
            sombra_sup = atual['high'] - max(atual['close'], atual['open'])
            sombra_inf = min(atual['close'], atual['open']) - atual['low']
            
            # Hammer (Martelo - Reversão de Alta)
            if sombra_inf > (corpo * 2) and sombra_sup < (corpo * 0.5):
                padroes.append("HAMMER")
            
            # Engulfing Bullish (Engolfo de Alta)
            if (anterior['close'] < anterior['open']) and (atual['close'] > atual['open']):
                if atual['open'] < anterior['close'] and atual['close'] > anterior['open']:
                    padroes.append("ENGULFING_BULL")

        except: pass
        return padroes