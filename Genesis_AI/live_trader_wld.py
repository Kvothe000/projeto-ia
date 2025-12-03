# Genesis_AI/live_trader_wld.py (INTEGRADO AO DASHBOARD)
import time
import pandas as pd
import numpy as np
import sys
import os
from stable_baselines3 import PPO

# Imports de Sistema
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from binance_connector import BinanceConnector
from manager import GerenciadorEstado
try:
    sys.path.append(current_dir)
    from features_engine import FeaturesEngine
except: pass

# CONFIG
MODELO_PATH = os.path.join(current_dir, "cerebros", "genesis_wld_veteran")
PAR_ALVO = "WLDUSDT"
TIMEFRAME = "15m"
WINDOW_SIZE = 30

# Catraca de Lucro (Trailing Stop)
DEGRAUS_PROFIT = [
    (0.005, 0.001), # 0.5% -> Garante 0.1%
    (0.010, 0.005), # 1.0% -> Garante 0.5%
    (0.015, 0.010), # 1.5% -> Garante 1.0%
    (0.025, 0.020), # 2.5% -> Garante 2.0%
]

def main():
    print(f"🤖 GÊNESIS WLD (LIVE & LOGGING)...")
    
    try:
        con = BinanceConnector()
        # Inicializa (se não tiver saldo, começa com 200)
        gerenciador = GerenciadorEstado(saldo_inicial=200.0)
        
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

    em_posicao = False
    preco_entrada = 0
    lado_trade = None 
    max_pnl = -1.0
    capital_reservado = 0
    
    saldo_atual = gerenciador.obter_saldo_disponivel()
    print(f"💰 Cofre: ${saldo_atual:.2f}")

    while True:
        try:
            time.sleep(2)
            
            # Dados
            df_raw = con.buscar_candles(PAR_ALVO, TIMEFRAME, limit=200)
            df_btc = con.buscar_candles("BTCUSDT", TIMEFRAME, limit=200)
            if df_raw is None or df_btc is None: continue

            # Features
            df_proc = FeaturesEngine.processar_dados(df_raw, df_btc)
            if df_proc is None or len(df_proc) < WINDOW_SIZE: continue

            # Observação
            for col in COLS_TREINO:
                if col not in df_proc.columns: df_proc[col] = 0
            df_feat = df_proc[COLS_TREINO].copy()
            df_norm = (df_feat - global_mean) / global_std
            df_norm = df_norm.fillna(0).clip(-5, 5)
            obs = df_norm.tail(WINDOW_SIZE).values.flatten()

            # Decisão
            action, _ = model.predict(obs, deterministic=True)
            action = action.item()
            
            preco_atual = df_raw.iloc[-1]['close']
            sinal = ["NEUTRO", "BUY", "SELL", "CLOSE"][action]

            # Dashboard Update
            import pandas_ta as ta
            df_raw.ta.adx(length=14, append=True)
            adx = df_raw.iloc[-1].get('ADX_14', 0)
            gerenciador.atualizar_monitor([{
                "par": PAR_ALVO, "preco": preco_atual, "adx": round(adx, 1),
                "sinal": sinal, "confianca": 100 if action else 0, "status_adx": "LIVE"
            }])

            # --- GESTÃO ---
            if em_posicao:
                pnl_pct = (preco_atual - preco_entrada) / preco_entrada
                if lado_trade == 2: pnl_pct = -pnl_pct
                if pnl_pct > max_pnl: max_pnl = pnl_pct
                
                # Trailing
                stop_dinamico = -0.015
                for gatilho, stop in DEGRAUS_PROFIT:
                    if max_pnl >= gatilho: stop_dinamico = stop
                
                stopou = pnl_pct <= stop_dinamico
                pnl_usd = (capital_reservado * 5) * pnl_pct
                
                cor = "\033[92m" if pnl_usd > 0 else "\033[91m"
                print(f"🛡️ WLD: {cor}${pnl_usd:.2f} ({pnl_pct*100:.2f}%){'\033[0m'} | Max: {max_pnl*100:.2f}% | Stop: {stop_dinamico*100:.1f}%")

                # Saída
                sair = False
                motivo = ""
                if action == 3: sair = True; motivo = "IA (Close)"
                elif stopou: sair = True; motivo = f"Trailing ({stop_dinamico*100:.1f}%)"
                elif action == 0: sair = True; motivo = "Sinal Neutro" # Sai se perder força
                elif (action == 1 and lado_trade == 2) or (action == 2 and lado_trade == 1):
                    sair = True; motivo = "Inversão"

                if sair:
                    print(f"👋 FECHANDO: {motivo}")
                    custo = (capital_reservado * 5) * 0.0012
                    liquido = capital_reservado + pnl_usd - custo
                    
                    # Devolve e Grava
                    gerenciador.devolver_capital(liquido)
                    gerenciador.registrar_trade(
                        PAR_ALVO, "CLOSE", preco_atual, 0, 0, 
                        tipo=motivo, 
                        pnl_usd=pnl_usd-custo,  # Lucro Líquido para o Gráfico
                        pnl_pct=pnl_pct*100
                    )
                    
                    print(f"💰 Resultado: ${pnl_usd-custo:.2f} | Novo Saldo: ${liquido:.2f}")
                    em_posicao = False
                    capital_reservado = 0

            # --- ENTRADA ---
            if not em_posicao and action in [1, 2]:
                if gerenciador.pode_enviar_alerta(PAR_ALVO, TIMEFRAME):
                    capital = gerenciador.reservar_capital()
                    if capital > 0:
                        tipo = "BUY" if action == 1 else "SELL"
                        print(f"\n🚀 ENTRADA: {tipo} | Capital: ${capital:.2f}")
                        em_posicao = True
                        lado_trade = action
                        preco_entrada = preco_atual
                        capital_reservado = capital
                        max_pnl = -0.001
                        
                        gerenciador.registrar_envio(PAR_ALVO)
                        gerenciador.registrar_trade(PAR_ALVO, tipo, preco_atual, 0, capital, "OPEN")
            
            if not em_posicao:
                cor = "\033[93m"
                if action == 1: cor = "\033[92m"
                elif action == 2: cor = "\033[91m"
                print(f"👀 WLD: {cor}{sinal}{'\033[0m'} | Preço: {preco_atual}")

            time.sleep(10)

        except KeyboardInterrupt: break
        except Exception as e: print(f"❌ Erro: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()