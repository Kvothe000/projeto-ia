# Genesis_AI/live_trader_wld.py (COM SINCRONIZAÇÃO REAL BINANCE)
import time
import pandas as pd
import numpy as np
import sys
import os
from stable_baselines3 import PPO

# Imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from binance_connector import BinanceConnector
from manager import GerenciadorEstado

try:
    from features_engine import FeaturesEngine
except ImportError:
    sys.path.append(current_dir)
    from features_engine import FeaturesEngine

# CONFIG
MODELO_NOME = "genesis_wld_v2" 
MODELO_PATH = os.path.join(current_dir, "cerebros", MODELO_NOME)
PAR_ALVO = "WLDUSDT"
TIMEFRAME = "15m"
WINDOW_SIZE = 30 

# Trailing Stop
DEGRAUS_PROFIT = [
    (0.005, 0.001), (0.010, 0.005), (0.015, 0.010), (0.025, 0.020)
]

def main():
    print(f"🚨 GÊNESIS WLD (SINCRONIZADO COM BINANCE) 🚨")
    
    try:
        con = BinanceConnector()
        
        # Sincroniza Saldo Inicial
        saldo_binance = con.obter_saldo_usdt()
        print(f"🏦 Saldo Binance: ${saldo_binance:.2f}")
        gerenciador = GerenciadorEstado(saldo_inicial=saldo_binance)
        
        # Carrega Modelo
        path = MODELO_PATH + ".zip" if not os.path.exists(MODELO_PATH) else MODELO_PATH
        if not os.path.exists(path): print(f"❌ Modelo não achado."); return
        model = PPO.load(path)
        
        # Normalização
        df_ref = pd.read_csv(os.path.join(parent_dir, "dataset_wld_clean.csv"))
        df_num = df_ref.select_dtypes(include=[np.number])
        cols_drop = ['target', 'timestamp', 'close']
        df_clean = df_num.drop(columns=[c for c in cols_drop if c in df_num.columns])
        COLS_TREINO = df_clean.columns.tolist()
        global_mean = df_clean.mean(); global_std = df_clean.std()

    except Exception as e: print(f"❌ Erro init: {e}"); return

    # Estado (Será sobrescrito pela Binance)
    em_posicao = False
    preco_entrada = 0
    lado_trade = None 
    max_pnl = -1.0
    capital_reservado = 0
    qtd_posicao = 0 
    
    print("🔭 Sincronizando e Monitorando...")

    while True:
        try:
            time.sleep(2)
            
            # --- 0. SINCRONIZAÇÃO COM A REALIDADE (A CORREÇÃO) ---
            posicao_real = con.obter_posicao_atual(PAR_ALVO)
            
            if posicao_real:
                # Se a Binance diz que temos posição, nós temos!
                if not em_posicao:
                    print(f"⚠️ DETECTADA POSIÇÃO EXISTENTE: {posicao_real['qtd']} moedas a {posicao_real['preco_entrada']}")
                    # Assume a posição
                    em_posicao = True
                    lado_trade = posicao_real['lado']
                    preco_entrada = posicao_real['preco_entrada']
                    qtd_posicao = posicao_real['qtd']
                    # Estima capital investido para cálculos
                    capital_reservado = (qtd_posicao * preco_entrada) / 5 # Assumindo 5x
                    
                    # Recalcula max_pnl baseado no preço atual para não estopar errado
                    preco_atual_sync = con.obter_preco_atual(PAR_ALVO)
                    pnl_pct_sync = (preco_atual_sync - preco_entrada) / preco_entrada
                    if lado_trade == 2: pnl_pct_sync = -pnl_pct_sync
                    max_pnl = max(max_pnl, pnl_pct_sync)
            else:
                # Se a Binance diz que não temos, então não temos!
                if em_posicao:
                    print("⚠️ Posição fechada externamente ou liquidada. Resetando estado.")
                    em_posicao = False
                    qtd_posicao = 0
                    capital_reservado = 0
                    max_pnl = -1.0
            # -----------------------------------------------------

            # 1. Coleta Dados
            df_raw = con.buscar_candles(PAR_ALVO, TIMEFRAME, limit=200)
            df_btc = con.buscar_candles("BTCUSDT", TIMEFRAME, limit=200)
            if df_raw is None or df_btc is None: continue

            # 2. Features
            df_proc = FeaturesEngine.processar_dados(df_raw, df_btc)
            if df_proc is None or len(df_proc) < WINDOW_SIZE: continue

            # 3. Prep IA
            for col in COLS_TREINO:
                if col not in df_proc.columns: df_proc[col] = 0
            df_feat = df_proc[COLS_TREINO].copy()
            df_norm = (df_feat - global_mean) / global_std
            df_norm = df_norm.fillna(0).clip(-5, 5)
            obs = df_norm.tail(WINDOW_SIZE).values.flatten()

            # 4. Decisão
            action, _ = model.predict(obs, deterministic=True)
            action = action.item()
            
            preco_atual = df_raw.iloc[-1]['close']
            sinal = ["NEUTRO", "BUY", "SELL", "CLOSE"][action]
            
            # Dashboard
            import pandas_ta as ta
            df_raw.ta.adx(length=14, append=True)
            adx = df_raw.iloc[-1].get('ADX_14', 0)
            gerenciador.atualizar_monitor([{
                "par": PAR_ALVO, "preco": preco_atual, "adx": round(adx, 1),
                "sinal": sinal, "confianca": 100 if action else 0, "status_adx": "REAL-SYNC"
            }])

            # --- GESTÃO DE POSIÇÃO ---
            if em_posicao:
                pnl_pct = (preco_atual - preco_entrada) / preco_entrada
                if lado_trade == 2: pnl_pct = -pnl_pct
                if pnl_pct > max_pnl: max_pnl = pnl_pct
                
                stop_dinamico = -0.02 
                for gatilho, stop in DEGRAUS_PROFIT:
                    if max_pnl >= gatilho: stop_dinamico = stop
                
                stopou = pnl_pct <= stop_dinamico
                
                # Usa o PnL Real da API se disponível, senão estima
                pnl_usd_real = posicao_real['pnl_usd'] if posicao_real else ((qtd_posicao * preco_atual) - (qtd_posicao * preco_entrada))
                if lado_trade == 2 and not posicao_real: pnl_usd_real = -pnl_usd_real
                
                cor = "\033[92m" if pnl_pct > 0 else "\033[91m"
                print(f"🛡️ WLD: {cor}${pnl_usd_real:.2f} ({pnl_pct*100:.2f}%){'\033[0m'} | Stop: {stop_dinamico*100:.1f}%")

                sair = False
                motivo = ""
                if action == 3: sair = True; motivo = "IA (Close)"
                elif stopou: sair = True; motivo = f"Trailing ({stop_dinamico*100:.1f}%)"
                elif action == 0: sair = True; motivo = "Sinal Neutro"
                elif (action == 1 and lado_trade == 2) or (action == 2 and lado_trade == 1):
                    sair = True; motivo = "Inversão IA"

                if sair:
                    print(f"👋 FECHANDO REAL: {motivo}")
                    con.cancelar_todas_ordens(PAR_ALVO)
                    
                    lado_saida = "SELL" if lado_trade == 1 else "BUY"
                    
                    # Usa a quantidade exata da Binance
                    qtd_saida = posicao_real['qtd'] if posicao_real else qtd_posicao
                    
                    res = con.client.futures_create_order(symbol=PAR_ALVO, side=lado_saida, type='MARKET', quantity=qtd_saida)
                    
                    if res and 'status' in res and res['status'] == 'FILLED':
                        preco_saida = float(res['avgPrice'])
                        print(f"   ✅ Saída Executada: {preco_saida}")
                        
                        # Atualiza Manager com Saldo Real
                        saldo_final = con.obter_saldo_usdt()
                        gerenciador.devolver_capital(saldo_final)
                        
                        gerenciador.registrar_trade(PAR_ALVO, "CLOSE", preco_saida, qtd_saida, 0, tipo=motivo, pnl_usd=pnl_usd_real, pnl_pct=pnl_pct*100)
                        em_posicao = False
                        max_pnl = -1.0
                    else:
                        print(f"❌ Erro Saída: {res}")

            # --- ENTRADA ---
            if not em_posicao and action in [1, 2]:
                if gerenciador.pode_enviar_alerta(PAR_ALVO, TIMEFRAME):
                    # Sincroniza saldo antes de entrar
                    saldo_disp = con.obter_saldo_usdt()
                    # Reserva lógica no manager (só para log)
                    gerenciador.reservar_capital() 
                    
                    if saldo_disp > 10:
                        tipo = "BUY" if action == 1 else "SELL"
                        print(f"\n🚀 ENTRADA: {tipo} | Saldo: ${saldo_disp:.2f}")
                        
                        poder_fogo = (saldo_disp * 0.98) * 5 
                        qtd = con.calcular_qtd_correta(PAR_ALVO, poder_fogo, preco_atual)
                        
                        if qtd > 0:
                            # Limpa ordens velhas
                            con.cancelar_todas_ordens(PAR_ALVO)
                            
                            ordem = con.client.futures_create_order(symbol=PAR_ALVO, side=tipo, type='MARKET', quantity=qtd)
                            
                            if ordem and 'status' in ordem and ordem['status'] == 'FILLED':
                                preco_exec = float(ordem['avgPrice'])
                                print(f"   ✅ Executado: {preco_exec}")
                                
                                # Stop Loss Binance
                                preco_stop = preco_exec * (0.98 if tipo == "BUY" else 1.02)
                                lado_stop = "SELL" if tipo == "BUY" else "BUY"
                                try: con.colocar_stop_loss(PAR_ALVO, lado_stop, qtd, round(preco_stop, 4))
                                except: pass
                                
                                em_posicao = True
                                lado_trade = action
                                preco_entrada = preco_exec
                                qtd_posicao = qtd
                                max_pnl = -0.001
                                
                                gerenciador.registrar_envio(PAR_ALVO)
                                gerenciador.registrar_trade(PAR_ALVO, tipo, preco_exec, qtd, poder_fogo, "OPEN-REAL")
                            else:
                                print("❌ Falha execução.")
                    else:
                        print(f"⏳ Saldo insuficiente: ${saldo_disp:.2f}")

            if not em_posicao:
                cor = "\033[93m"
                if action == 1: cor = "\033[92m"
                elif action == 2: cor = "\033[91m"
                print(f"👀 WLD: {cor}{sinal}{'\033[0m'} | Preço: {preco_atual}")

            time.sleep(10)

        except KeyboardInterrupt:
            print("\n🛑 Parando..."); break
        except Exception as e:
            print(f"❌ Erro Loop: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()