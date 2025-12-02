# Genesis_AI/live_trader_TEMPLATE.py 
# (Salve como live_trader_pepe.py e mude PAR_ALVO para '1000PEPEUSDT' e MODELO para 'genesis_pepe_v1')
# (Salve como live_trader_wld.py e mude PAR_ALVO para 'WLDUSDT' e MODELO para 'genesis_wld_v1')

import time
import pandas as pd
import numpy as np
import sys
import os
from stable_baselines3 import PPO

# --- CONFIGURAÇÕES ESPECÍFICAS (MUDE AQUI) ---
PAR_ALVO = "1000PEPEUSDT" # ou "WLDUSDT"
MODELO_NOME = "genesis_pepe_v1" # ou "genesis_wld_v1"
DATASET_TREINO = "dataset_pepe_clean.csv" # ou "dataset_wld_clean.csv"
# ----------------------------------------------

TIMEFRAME = "15m"
WINDOW_SIZE = 50
ALAVANCAGEM = 5

# --- A CATRACA DE LUCRO (Trailing Stop) ---
# Formato: (Lucro Atingido, Stop Garantido)
# Ex: Se bater 1.2%, garante 0.5%. Se bater 2%, garante 1.5%.
DEGRAUS_PROFIT = [
    (0.008, 0.002), # Nível 1: Pagou taxas (0.8% -> 0.2%)
    (0.012, 0.006), # Nível 2: Meta Mínima (1.2% -> 0.6%)
    (0.020, 0.012), # Nível 3: Surfando (2.0% -> 1.2%)
    (0.040, 0.030), # Nível 4: Lua (4.0% -> 3.0%)
]

# Imports de Sistema
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from binance_connector import BinanceConnector
from manager import GerenciadorEstado
try:
    sys.path.append(current_dir)
    from features_engine import FeaturesEngine
    from memory_system import MemorySystem
except: pass

def main():
    print(f"🤖 LIVE TRADER INICIADO: {PAR_ALVO}")
    
    try:
        con = BinanceConnector()
        gerenciador = GerenciadorEstado()
        memoria = MemorySystem()
        
        # 1. Cérebro
        modelo_path = os.path.join(current_dir, "cerebros", MODELO_NOME)
        if not os.path.exists(modelo_path + ".zip"):
            print(f"❌ Modelo {MODELO_NOME} não encontrado!"); return
        model = PPO.load(modelo_path)
        
        # 2. Normalização (Do Treino)
        df_ref_path = os.path.join(parent_dir, DATASET_TREINO)
        if os.path.exists(df_ref_path):
            df_ref = pd.read_csv(df_ref_path)
            df_num = df_ref.select_dtypes(include=[np.number])
            cols_drop = ['target', 'timestamp', 'close']
            df_clean = df_num.drop(columns=[c for c in cols_drop if c in df_num.columns])
            COLS_TREINO = df_clean.columns.tolist()
            global_mean = df_clean.mean()
            global_std = df_clean.std()
        else:
            print("⚠️ Dataset treino não achado. Usando normalização local.")
            return

    except Exception as e: print(f"❌ Erro init: {e}"); return

    # Estado Local
    em_posicao = False
    preco_entrada = 0
    lado_trade = None 
    max_pnl = -1.0
    capital_reservado = 0

    print(f"🔭 {PAR_ALVO}: À espera de capital e oportunidade...")

    while True:
        try:
            time.sleep(2)
            
            # Coleta
            df_raw = con.buscar_candles(PAR_ALVO, TIMEFRAME, limit=200)
            df_btc = con.buscar_candles("BTCUSDT", TIMEFRAME, limit=200)
            if df_raw is None or df_btc is None: continue

            # Processa
            df_proc = FeaturesEngine.processar_dados(df_raw, df_btc)
            if df_proc is None or len(df_proc) < WINDOW_SIZE: continue

            # Prepara IA
            for col in COLS_TREINO:
                if col not in df_proc.columns: df_proc[col] = 0
            df_feat = df_proc[COLS_TREINO].copy()
            df_norm = (df_feat - global_mean) / global_std
            df_norm = df_norm.fillna(0).clip(-5, 5)
            
            obs = df_norm.tail(WINDOW_SIZE).values.flatten()
            
            # Decisão IA
            action, _ = model.predict(obs, deterministic=True)
            action = action.item()
            
            # Dashboard Update
            preco_atual = df_raw.iloc[-1]['close']
            sinal_str = ["NEUTRO", "BUY", "SELL", "CLOSE"][action]
            
            gerenciador.atualizar_monitor([{
                "par": PAR_ALVO, "preco": preco_atual, "adx": 0,
                "sinal": sinal_str, "confianca": 100 if action else 0, "status_adx": "ACTIVE"
            }])

            # --- GESTÃO DE POSIÇÃO ---
            if em_posicao:
                # PnL
                pnl_pct = (preco_atual - preco_entrada) / preco_entrada
                if lado_trade == 2: pnl_pct = -pnl_pct
                
                # Atualiza Máximo
                if pnl_pct > max_pnl: max_pnl = pnl_pct
                
                # Catraca (Trailing)
                stop_dinamico = -0.015 # Stop Loss inicial de emergência (-1.5%)
                for gatilho, stop in DEGRAUS_PROFIT:
                    if max_pnl >= gatilho: stop_dinamico = stop
                
                stopou = pnl_pct <= stop_dinamico
                
                pnl_usd = (capital_reservado * ALAVANCAGEM) * pnl_pct
                cor = "\033[92m" if pnl_pct > 0 else "\033[91m"
                status_stop = f"Stop: {stop_dinamico*100:.1f}%"
                
                print(f"🛡️ {PAR_ALVO}: {cor}${pnl_usd:.2f} ({pnl_pct*100:.2f}%){'\033[0m'} | Max: {max_pnl*100:.2f}% | {status_stop}")

                # SAÍDA
                sair = False
                motivo = ""
                if action == 3: sair = True; motivo = "IA (Close)"
                elif stopou: sair = True; motivo = f"Trailing ({stop_dinamico*100:.1f}%)"
                # Inversão de mão também sai
                elif (action == 1 and lado_trade == 2) or (action == 2 and lado_trade == 1):
                    sair = True; motivo = "Inversão IA"

                if sair:
                    print(f"👋 FECHANDO {PAR_ALVO}: {motivo}")
                    
                    # Taxas estimadas
                    custo = (capital_reservado * ALAVANCAGEM) * 0.0012 
                    liquido = capital_reservado + pnl_usd - custo
                    
                    # Devolve ao Cofre
                    novo_total = gerenciador.devolver_capital(liquido)
                    print(f"💰 Devolvido: ${liquido:.2f} | Cofre Total: ${novo_total:.2f}")
                    
                    # Memoriza Experiência
                    memoria.memorizar(df_features=df_feat.tail(1), action=lado_trade, reward=pnl_pct, pnl_pct=pnl_pct)
                    
                    gerenciador.registrar_trade(PAR_ALVO, "CLOSE", preco_atual, 0, 0, "CLOSE", pnl_usd, pnl_pct*100)
                    
                    em_posicao = False
                    capital_reservado = 0

            # --- ENTRADA ---
            if not em_posicao and action in [1, 2]:
                if gerenciador.pode_enviar_alerta(PAR_ALVO, TIMEFRAME):
                    
                    # 1. TENTA PEGAR O DINHEIRO
                    capital = gerenciador.reservar_capital()
                    
                    if capital > 0:
                        tipo = "BUY" if action == 1 else "SELL"
                        print(f"\n🚀 ENTRADA {PAR_ALVO}: {tipo} | Capital: ${capital:.2f}")
                        
                        # 2. Executa (Simulação/Real)
                        # con.colocar_ordem...
                        
                        em_posicao = True
                        lado_trade = action
                        preco_entrada = preco_atual
                        capital_reservado = capital
                        max_pnl = -0.001
                        
                        gerenciador.registrar_envio(PAR_ALVO)
                        gerenciador.registrar_trade(PAR_ALVO, tipo, preco_atual, 0, capital, "OPEN")
                    else:
                        print(f"⏳ {PAR_ALVO} quer entrar, mas Cofre vazio (Outra moeda operando?)")

            if not em_posicao:
                saldo = gerenciador.obter_saldo_disponivel()
                cor = "\033[93m"
                if action == 1: cor = "\033[92m"
                elif action == 2: cor = "\033[91m"
                print(f"👀 {PAR_ALVO}: {cor}{sinal_str}{'\033[0m'} | Cofre Livre: ${saldo:.2f}")

            time.sleep(10)

        except KeyboardInterrupt: break
        except Exception as e: print(f"❌ Erro: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()