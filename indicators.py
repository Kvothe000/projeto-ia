# indicators.py (Versão Final Compatível v16.1)
import pandas as pd
import pandas_ta as ta
import config

class Calculadora:
    @staticmethod
    def adicionar_todos(df):
        if len(df) < 20: return df
        try:
            df.ta.atr(length=14, append=True)
            df.ta.adx(length=14, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.kc(length=20, scalar=1.5, append=True)
            df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            
            df['EMA_9'] = df.ta.ema(close=df['close'], length=9)
            df['EMA_21'] = df.ta.ema(close=df['close'], length=21)
            df['EMA_200'] = df.ta.ema(close=df['close'], length=200)
            df['VOL_SMA_20'] = df.ta.sma(close=df['volume'], length=20)
            
            df['EMA_21'] = df['EMA_21'].fillna(df['close']) 
            df['DIST_EMA21'] = (df['close'] - df['EMA_21']) / df['EMA_21'] * 100
        except Exception as e:
            print(f"Erro Calc: {e}")
        return df

class Estrategia:
    @staticmethod
    def calcular_posicao_e_risco(candle, saldo_disponivel, direcao="LONG", modo_operacao="SPOT"):
        atr = candle.get('ATR_14')
        if pd.isna(atr) or atr is None: return None
        preco = candle['close']
        
        fator_rr = config.RR_FUTUROS if modo_operacao == "FUTUROS" else config.RR_SPOT
        mult_trailing = 1.5 if modo_operacao == "FUTUROS" else 3.0

        if direcao == "LONG":
            stop_loss = preco - (2.0 * atr)
            take_profit = preco + (abs(preco - stop_loss) * fator_rr)
            trailing_stop = preco - (mult_trailing * atr)
        else:
            stop_loss = preco + (2.0 * atr)
            take_profit = preco - (abs(preco - stop_loss) * fator_rr)
            trailing_stop = preco + (mult_trailing * atr)
        
        distancia_stop = abs(preco - stop_loss)
        stop_perc = (distancia_stop / preco) * 100
        
        lev_segura = 1
        if modo_operacao == "FUTUROS" and stop_perc > 0:
            lev_segura = int(20 / stop_perc) 
            lev_segura = min(max(lev_segura, 1), 10)
        
        risco_usd = saldo_disponivel * (config.RISCO_POR_TRADE / 100)
        if distancia_stop == 0: return None
        qtd_moedas = risco_usd / distancia_stop
        total_usd_posicao = qtd_moedas * preco
        
        return {
            "stop": stop_loss, "tp": take_profit, "stop_perc": stop_perc, 
            "qtd": qtd_moedas, "total": total_usd_posicao, "alavancagem": lev_segura,
            "trailing": trailing_stop, "direcao": direcao, "multiplicador_trailing": mult_trailing
        }

    @staticmethod
    def obter_regime(candle):
        try:
            bb_up = candle.get('BBU_20_2.0')
            bb_low = candle.get('BBL_20_2.0')
            kc_up = candle.get('KCUe_20_1.5')
            kc_low = candle.get('KCLe_20_1.5')
            adx = candle.get('ADX_14', 0)

            if any(x is None for x in [bb_up, bb_low, kc_up, kc_low]): return 'LATERAL'
            if (bb_up < kc_up) and (bb_low > kc_low): return 'SQUEEZE'
            if adx > 25: return 'TENDENCIA'
        except: pass
        return 'LATERAL'

    @staticmethod
    def verificar_exaustao(candle):
        dist = candle.get('DIST_EMA21')
        if dist is not None and not pd.isna(dist):
            return abs(dist) > config.MAX_DISTANCIA_MEDIA_PCT
        return False

    @staticmethod
    # AQUI ESTAVA O ERRO: Adicionei **kwargs para aceitar qualquer argumento extra (como df_sup) sem quebrar
    def analisar_sinais(df, *args, **kwargs):
        if len(df) < 5: return [], 'INDEFINIDO', None
            
        c_atual = df.iloc[-1]
        c_ant = df.iloc[-2]
        msgs = []
        direcao = None
        regime = Estrategia.obter_regime(c_ant)
        
        if Estrategia.verificar_exaustao(c_atual):
            return [], regime, None

        def get_val(candle, key, default=0): return candle.get(key, default)

        close = c_atual['close']
        ema200 = get_val(c_atual, 'EMA_200')
        ema21 = get_val(c_ant, 'EMA_21')
        bbu = get_val(c_atual, 'BBU_20_2.0')
        bbl = get_val(c_atual, 'BBL_20_2.0')
        vol = get_val(c_atual, 'volume')
        vol_sma = get_val(c_atual, 'VOL_SMA_20')

        # SINAIS LONG
        if regime == 'SQUEEZE':
            if close > bbu and vol > vol_sma:
                msgs.append("💣 LONG: TTM SQUEEZE")
                direcao = 'LONG'
        elif regime == 'TENDENCIA' and close > ema200:
            if get_val(c_ant, 'low') <= ema21 and get_val(c_ant, 'close') > ema21:
                msgs.append("📈 LONG: Pullback EMA 21")
                direcao = 'LONG'
        elif regime == 'LATERAL':
            bbl_ant = get_val(c_ant, 'BBL_20_2.0')
            if get_val(c_ant, 'low') <= bbl_ant and get_val(c_ant, 'close') > bbl_ant:
                k = get_val(c_ant, 'STOCHk_14_3_3')
                d = get_val(c_ant, 'STOCHd_14_3_3')
                if k < 30 and k > d:
                    msgs.append("🔄 LONG: Reversão Fundo")
                    direcao = 'LONG'

        # SINAIS SHORT
        if not direcao:
            if regime == 'SQUEEZE':
                if close < bbl and vol > vol_sma:
                    msgs.append("🔻 SHORT: TTM SQUEEZE")
                    direcao = 'SHORT'
            elif regime == 'TENDENCIA' and close < ema200:
                if get_val(c_ant, 'high') >= ema21 and get_val(c_ant, 'close') < ema21:
                    msgs.append("📉 SHORT: Pullback EMA 21")
                    direcao = 'SHORT'
            elif regime == 'LATERAL':
                bbu_ant = get_val(c_ant, 'BBU_20_2.0')
                if get_val(c_ant, 'high') >= bbu_ant and get_val(c_ant, 'close') < bbu_ant:
                    k = get_val(c_ant, 'STOCHk_14_3_3')
                    d = get_val(c_ant, 'STOCHd_14_3_3')
                    if k > 70 and k < d:
                        msgs.append("🔄 SHORT: Reversão Topo")
                        direcao = 'SHORT'

        return msgs, regime, direcao