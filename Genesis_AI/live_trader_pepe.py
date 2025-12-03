# Genesis_AI/live_trader_pepe.py (VERSÃO FINAL: LEOPARDO INTELIGENTE)
import time
import pandas as pd
import numpy as np
import sys
import os
import json
from stable_baselines3 import PPO

# Imports de Sistema
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

# --- CONFIGURAÇÃO PEPE ---
MODELO_NOME = "genesis_pepe_v1"
MODELO_PATH = os.path.join(current_dir, "cerebros", MODELO_NOME)
PAR_ALVO = "1000PEPEUSDT"
TIMEFRAME = "15m"
WINDOW_SIZE = 50  # <--- PEPE TREINOU COM 50!
ARQUIVO_POSICAO = f"posicao_{PAR_ALVO}.json"

# TRAILING STOP (Ajustado para volatilidade da PEPE)
# A PEPE mexe mais, então os degraus são ligeiramente mais espaçados
DEGRAUS_PROFIT = [
    (0.006, 0.001), # 0.6% -> Garante taxas
    (0.012, 0.005), # 1.2% -> Garante 0.5%
    (0.020, 0.010), # 2.0% -> Garante 1.0%
    (0.030, 0.020), # 3.0% -> Garante 2.0%
    (0.050, 0.035), # 5.0% -> Garante 3.5% (Lua!)
]

def salvar_estado_local(em_posicao, preco_entrada, lado_trade, max_pnl, valor_inv):
    dados = {
        "em_posicao": em_posicao,
        "preco_entrada": preco_entrada,
        "lado_trade": lado_trade,
        "max_pnl": max_pnl,
        "valor_investido": valor_inv
    }
    try:
        with open(ARQUIVO_POSICAO, 'w') as f:
            json.dump(dados, f)
    except: pass

def carregar_estado_local():
    if os.path.exists(ARQUIVO_POSICAO):
        try:
            with open(ARQUIVO_POSICAO, 'r') as f:
                return json.load(f)
        except: pass
    return None

def main():
    print(f"🐆 LEOPARDO PEPE INICIADO ({PAR_ALVO})...")
    
    try:
        con = BinanceConnector()
        gerenciador = GerenciadorEstado(saldo_inicial=200.0)
        
        # Modelo
        path = MODELO_PATH + ".zip" if not os.path.exists(MODELO_PATH) else MODELO_PATH
        if not os.path.exists(path):
            print(f"❌ Erro: Modelo {MODELO_NOME} não encontrado.")
            return
        
        model = PPO.load(path)
        print("🧠 Cérebro PEPE Carregado!")
        
        # Normalização (Carrega do Treino da PEPE)
        df_ref_path = os.path.join(parent_dir, "dataset_pepe_clean.csv")
        if os.path.exists(df_ref_path):
            df_ref = pd.read_csv(df_ref_path)
            df_num = df_ref.select_dtypes(include=[np.number])
            cols_drop = ['target', 'timestamp', 'close']
            df_clean = df_num.drop(columns=[c for c in cols_drop if c in df_num.columns])
            COLS_TREINO = df_clean.columns.tolist()
            global_mean = df_clean.mean()
            global_std = df_clean.std()
            print(f"📊 Visão Sincronizada ({len(COLS_TREINO)} features).")
        else:
            print("⚠️ Aviso: Dataset PEPE não encontrado. Usando normalização local.")
            global_mean, global_std, COLS_TREINO = None, None, None

        # RECUPERA ESTADO
        state = carregar_estado_local()
        if state and state.get("em_posicao"):
            print(f"🔄 Restaurando trade PEPE aberto em {state['preco_entrada']}...")
            em_posicao = True
            preco_entrada = state["preco_entrada"]
            lado_trade = state["lado_trade"]
            max_pnl = state["max_pnl"]
            valor_investido = state["valor_investido"]
        else:
            em_posicao = False
            preco_entrada = 0
            lado_trade = None
            max_pnl = -1.0
            valor_investido = 0

    except Exception as e: print(f"❌ Erro init: {e}"); return

    saldo_atual = gerenciador.obter_saldo_disponivel()
    print(f"💰 Cofre Disponível: ${saldo_atual:.2f}")

    while True:
        try:
            time.sleep(2)
            
            # Coleta
            df_raw = con.buscar_candles(PAR_ALVO, TIMEFRAME, limit=200)
            df_btc = con.buscar_candles("BTCUSDT", TIMEFRAME, limit=200)
            if df_raw is None or df_btc is None: continue

            # Processamento
            df_proc = FeaturesEngine.processar_dados(df_raw, df_btc)
            if df_proc is None or len(df_proc) < WINDOW_SIZE: continue

            # Prepara IA
            if COLS_TREINO:
                for col in COLS_TREINO:
                    if col not in df_proc.columns: df_proc[col] = 0
                df_features = df_proc[COLS_TREINO].copy()
            else:
                # Fallback
                cols_ignore = ['timestamp', 'close', 'target']
                df_numeric = df_proc.select_dtypes(include=[np.number])
                df_features = df_numeric.drop(columns=[c for c in cols_ignore if c in df_numeric.columns])

            # Normaliza
            if global_mean is not None:
                df_norm = (df_features - global_mean) / global_std
            else:
                df_norm = (df_features - df_features.mean()) / df_features.std()
            
            df_norm = df_norm.fillna(0).clip(-5, 5)
            obs = df_norm.tail(WINDOW_SIZE).values.flatten()

            # Decisão
            action, _ = model.predict(obs, deterministic=True)
            action = action.item()
            
            preco_atual = df_raw.iloc[-1]['close']
            sinal = ["NEUTRO", "BUY", "SELL", "CLOSE"][action]

            # Dashboard
            import pandas_ta as ta
            df_raw.ta.adx(length=14, append=True)
            adx_val = df_raw.iloc[-1].get('ADX_14', 0)
            gerenciador.atualizar_monitor([{
                "par": PAR_ALVO, "preco": preco_atual, "adx": round(adx_val, 1),
                "sinal": sinal, "confianca": 100 if action else 0, "status_adx": "GENESIS-PEPE"
            }])

            # --- GESTÃO DE POSIÇÃO ---
            if em_posicao:
                pnl_pct = (preco_atual - preco_entrada) / preco_entrada
                if lado_trade == 2: pnl_pct = -pnl_pct
                
                if pnl_pct > max_pnl:
                    max_pnl = pnl_pct
                    salvar_estado_local(True, preco_entrada, lado_trade, max_pnl, valor_investido)
                
                # Trailing Stop
                stop_dinamico = -0.025 # Stop Pânico para PEPE (2.5%)
                for gatilho, stop in DEGRAUS_PROFIT:
                    if max_pnl >= gatilho: stop_dinamico = stop
                
                stopou = pnl_pct <= stop_dinamico
                
                pnl_usd = (valor_investido * 5) * pnl_pct
                cor = "\033[92m" if pnl_pct > 0 else "\033[91m"
                print(f"🛡️ PEPE: {cor}${pnl_usd:.2f} ({pnl_pct*100:.2f}%){'\033[0m'} | Max: {max_pnl*100:.2f}% | Stop: {stop_dinamico*100:.1f}%")

                # Saída
                sair = False
                motivo = ""
                if action == 3: sair = True; motivo = "IA (Close)"
                elif stopou: sair = True; motivo = f"Trailing ({stop_dinamico*100:.1f}%)"
                elif action == 0: sair = True; motivo = "Sinal Neutro"
                elif (action == 1 and lado_trade == 2) or (action == 2 and lado_trade == 1):
                    sair = True; motivo = "Inversão IA"

                if sair:
                    print(f"👋 FECHANDO PEPE: {motivo}")
                    
                    # Descomente para ordem real
                    # con.cancelar_todas_ordens(PAR_ALVO)
                    # con.colocar_ordem_market(PAR_ALVO, "SELL" if lado_trade==1 else "BUY", ...)
                    
                    custo = (valor_investido * 5) * 0.0012
                    liquido = valor_investido + pnl_usd - custo
                    
                    # Devolve ao Cofre
                    gerenciador.devolver_capital(liquido)
                    print(f"💰 Devolvido: ${liquido:.2f}")
                    
                    gerenciador.registrar_trade(PAR_ALVO, "CLOSE", preco_atual, 0, 0, "CLOSE", pnl_usd, pnl_pct*100)
                    
                    em_posicao = False
                    valor_investido = 0
                    salvar_estado_local(False, 0, 0, 0, 0)

            # --- ENTRADA ---
            if not em_posicao and action in [1, 2]:
                if gerenciador.pode_enviar_alerta(PAR_ALVO, TIMEFRAME):
                    
                    # Pede Dinheiro ao Cofre Central
                    capital = gerenciador.reservar_capital()
                    
                    if capital > 0:
                        tipo = "BUY" if action == 1 else "SELL"
                        print(f"\n🚀 PEPE ENTRADA: {tipo} | Capital: ${capital:.2f}")
                        
                        qtd = con.calcular_qtd_correta(PAR_ALVO, capital * 5, preco_atual)
                        if qtd > 0:
                            # con.colocar_ordem_market(...)
                            
                            em_posicao = True
                            lado_trade = action
                            preco_entrada = preco_atual
                            valor_investido = capital
                            max_pnl = -0.001
                            
                            salvar_estado_local(True, preco_entrada, lado_trade, max_pnl, valor_investido)
                            gerenciador.registrar_envio(PAR_ALVO)
                            gerenciador.registrar_trade(PAR_ALVO, tipo, preco_atual, qtd, capital * 5, "OPEN")
                    else:
                        # Se não tiver saldo, não entra
                        pass

            if not em_posicao:
                cor = "\033[93m"
                if action == 1: cor = "\033[92m"
                elif action == 2: cor = "\033[91m"
                print(f"🐸 {PAR_ALVO}: {cor}{sinal[action]}{'\033[0m'} | Preço: {preco_atual}")

            time.sleep(10)

        except KeyboardInterrupt:
            print("\n🛑 Parando..."); break
        except Exception as e:
            print(f"❌ Erro: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()