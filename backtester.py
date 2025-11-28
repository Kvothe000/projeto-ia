# backtester.py
import pandas as pd
from indicators import Estrategia

class Backtester:
    @staticmethod
    def simular_sinal_no_passado(df, tipo_sinal_atual):
        # Precisa de histórico decente
        if len(df) < 200: return 0, 0, 0
        
        trades_total = 0
        trades_win = 0
        saldo_acumulado_pct = 0.0
        
        # Taxa Binance (0.1% entrada + 0.1% saida = 0.2%)
        # Se você usar BNB para taxas, cai para 0.075% + 0.075% = 0.15%
        TAXA_TOTAL = 0.2 
        
        # Simula nos últimos 500 candles (ignorando os 20 mais recentes pra poder ver o futuro)
        inicio_analise = max(50, len(df) - 500)
        
        for i in range(inicio_analise, len(df) - 20):
            c_atual = df.iloc[i]
            c_ant = df.iloc[i-1]
            
            # Recria o sinal da época
            regime = Estrategia.obter_regime(c_ant)
            sinal_detectado = False
            
            # --- Lógica de Detecção (Simplificada para velocidade) ---
            if "SQUEEZE" in tipo_sinal_atual and regime == 'SQUEEZE':
                 if c_atual['close'] > c_atual['BBU_20_2.0']: sinal_detectado = True
            
            elif "Tendência" in tipo_sinal_atual and regime == 'TENDENCIA':
                if c_atual['close'] > c_atual['EMA_200']:
                    # Verifica cruzamento MACD simples
                    if c_ant['MACD_12_26_9'] > c_ant['MACDs_12_26_9'] and \
                       df.iloc[i-2]['MACD_12_26_9'] <= df.iloc[i-2]['MACDs_12_26_9']:
                        sinal_detectado = True
            
            elif "Reversão" in tipo_sinal_atual and regime == 'LATERAL':
                 if c_ant['low'] <= c_ant['BBL_20_2.0'] and c_ant['close'] > c_ant['BBL_20_2.0']:
                     sinal_detectado = True

            if sinal_detectado:
                dados = Estrategia.calcular_posicao_e_risco(c_atual)
                if not dados: continue
                
                stop_price = dados['stop']
                tp_price = dados['tp']
                
                trades_total += 1
                resultado_trade = 0 # 0 = Neutro, 1 = Win, -1 = Loss
                
                # Olha o futuro (Próximos 30 candles)
                for f in range(i+1, i+31):
                    c_fut = df.iloc[f]
                    
                    touched_stop = c_fut['low'] <= stop_price
                    touched_tp = c_fut['high'] >= tp_price
                    
                    # CENÁRIO PESSIMISTA (REALISTA):
                    # Se tocou no Stop e no TP no mesmo candle, assumimos LOSS.
                    # Motivo: O pânico (queda) geralmente acontece mais rápido que a euforia.
                    if touched_stop and touched_tp:
                        resultado_trade = -1
                        break
                    
                    if touched_stop:
                        resultado_trade = -1
                        break
                    
                    if touched_tp:
                        resultado_trade = 1
                        break
                
                # Se saiu do loop sem tocar em nada, é um "Time Stop" (fechou por tempo), consideramos neutro ou leve preju
                if resultado_trade == 1:
                    trades_win += 1
                    # Ganho líquido descontando taxas
                    lucro_bruto = ((tp_price - c_atual['close']) / c_atual['close']) * 100
                    saldo_acumulado_pct += (lucro_bruto - TAXA_TOTAL)
                elif resultado_trade == -1:
                    # Perda descontando taxas (aumenta o prejuizo)
                    perda_bruta = abs((stop_price - c_atual['close']) / c_atual['close']) * 100
                    saldo_acumulado_pct -= (perda_bruta + TAXA_TOTAL)

        if trades_total == 0: return 0, 0, 0
        
        win_rate = (trades_win / trades_total) * 100
        # Retorna Winrate, Total e Lucro/Prejuízo Acumulado na simulação
        return win_rate, trades_total, saldo_acumulado_pct