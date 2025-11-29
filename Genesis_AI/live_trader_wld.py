# Genesis_AI/live_trader_wld.py (IMPORT CORRIGIDO)
import time
import pandas as pd
import numpy as np
import sys
import os
from stable_baselines3 import PPO

# --- CORREÇÃO DE IMPORTAÇÃO ---
# Adiciona a pasta PAI (Binance) ao sistema para encontrar os módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
# ------------------------------

from binance_connector import BinanceConnector
from manager import GerenciadorEstado
# Importa o motor de features (pode estar na pasta atual ou na pai)
try:
    from features_engine import FeaturesEngine
except ImportError:
    # Tenta importar da pasta local Genesis_AI se não achar na pai
    sys.path.append(current_dir)
    from features_engine import FeaturesEngine

# --- CONFIGURAÇÃO ---
MODELO_PATH = os.path.join(current_dir, "cerebros", "genesis_wld_v1")
PAR_ALVO = "WLDUSDT"
TIMEFRAME = "15m"
CAPITAL_OPERACIONAL = 200  
ALAVANCAGEM = 5
WINDOW_SIZE = 30           

def main():
    print(f"🤖 INICIANDO LIVE TRADER GÊNESIS ({PAR_ALVO})...")
    
    try:
        con = BinanceConnector()
        gerenciador = GerenciadorEstado()
        
        # Carrega o Cérebro
        # Adiciona extensão .zip se necessário
        model_file = MODELO_PATH
        if not os.path.exists(model_file) and os.path.exists(model_file + ".zip"):
            model_file += ".zip"
            
        if not os.path.exists(model_file):
            print(f"❌ Erro: Modelo não encontrado em {MODELO_PATH}")
            return
        
        model = PPO.load(MODELO_PATH)
        print("🧠 Cérebro WLD Carregado e Pronto!")
        
    except Exception as e:
        print(f"❌ Falha na inicialização: {e}")
        import traceback
        traceback.print_exc()
        return

    em_posicao = False
    preco_entrada = 0
    lado_trade = None 
    
    print("🔭 Monitorando mercado em tempo real...")

    while True:
        try:
            time.sleep(2) 

            # 1. Coleta de Dados
            df_raw = con.buscar_candles(PAR_ALVO, TIMEFRAME, limit=100)
            if df_raw is None: continue
            
            df_btc = con.buscar_candles("BTCUSDT", TIMEFRAME, limit=100)
            if df_btc is None: continue

            # 2. Engenharia de Features
            df_proc = FeaturesEngine.processar_dados(df_raw, df_btc)
            
            if df_proc is None or len(df_proc) < WINDOW_SIZE:
                continue

            # 3. Prepara Observação
            cols_ia = FeaturesEngine.colunas_finais()
            
            # Normalização Z-Score (Simplificada para Live)
            df_features = df_proc[cols_ia].tail(100) 
            df_norm = (df_features - df_features.mean()) / df_features.std()
            df_norm = df_norm.fillna(0).clip(-5, 5)
            
            # Janela para a IA
            obs = df_norm.tail(WINDOW_SIZE).values.astype(np.float32)
            
            # Proteção de Shape
            if obs.shape != (WINDOW_SIZE, len(cols_ia)):
                print(f"⚠️ Shape incorreto: {obs.shape}")
                continue

            # 4. Decisão
            action, _ = model.predict(obs, deterministic=True)
            action = action.item()
            
            # Dados para Log
            preco_atual = df_raw.iloc[-1]['close']
            
            # Traduz Ação
            sinal_str = "NEUTRO"
            if action == 1: sinal_str = "BUY"
            elif action == 2: sinal_str = "SELL"
            elif action == 3: sinal_str = "CLOSE"

            cor = "\033[93m"
            if action == 1: cor = "\033[92m"
            elif action == 2: cor = "\033[91m"
            reset = "\033[0m"

            print(f"👀 {PAR_ALVO}: {cor}{sinal_str}{reset} | Preço: {preco_atual}")

            # Atualiza Dashboard
            status_dash = [{
                "par": PAR_ALVO,
                "preco": preco_atual,
                "adx": 0, # Opcional calcular
                "sinal": sinal_str,
                "confianca": 100 if action != 0 else 0,
                "status_adx": "GENESIS"
            }]
            gerenciador.atualizar_monitor(status_dash)

            # 5. Execução (Simulada por enquanto, descomente para Real)
            
            # FECHAR
            if em_posicao:
                sair = False
                if action == 3: sair = True
                elif action == 1 and lado_trade == 2: sair = True
                elif action == 2 and lado_trade == 1: sair = True
                
                if sair:
                    print(f"🛡️ FECHANDO POSIÇÃO EM {PAR_ALVO}...")
                    # con.colocar_ordem_market(PAR_ALVO, "SELL" if lado_trade==1 else "BUY", ...)
                    
                    lucro = (preco_atual - preco_entrada) / preco_entrada
                    if lado_trade == 2: lucro = -lucro
                    
                    gerenciador.registrar_trade(PAR_ALVO, "CLOSE", preco_atual, 0, CAPITAL_OPERACIONAL, f"Lucro: {lucro*100:.2f}%")
                    em_posicao = False
                    lado_trade = None

            # ABRIR
            if not em_posicao and action in [1, 2]:
                if gerenciador.pode_enviar_alerta(PAR_ALVO, TIMEFRAME):
                    tipo = "BUY" if action == 1 else "SELL"
                    print(f"\n🚀 GÊNESIS ORDENOU: {tipo}!")
                    
                    qtd = con.calcular_qtd_correta(PAR_ALVO, CAPITAL_OPERACIONAL, preco_atual)
                    
                    if qtd > 0:
                        print(f"💰 Entrada: {preco_atual} | Qtd: {qtd}")
                        # con.colocar_ordem_market(PAR_ALVO, tipo, qtd, ALAVANCAGEM)
                        
                        em_posicao = True
                        lado_trade = action
                        preco_entrada = preco_atual
                        
                        gerenciador.registrar_envio(PAR_ALVO)
                        gerenciador.registrar_trade(PAR_ALVO, tipo, preco_atual, qtd, CAPITAL_OPERACIONAL, "GENESIS-LIVE")

            time.sleep(10) 

        except KeyboardInterrupt:
            print("\n🛑 Gênesis Parado.")
            break
        except Exception as e:
            print(f"❌ Erro no loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()