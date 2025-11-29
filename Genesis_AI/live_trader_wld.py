# Genesis_AI/live_trader_wld.py (VERSÃO COM BUSCA AUTOMÁTICA DE DADOS)
import time
import pandas as pd
import numpy as np
import sys
import os
from stable_baselines3 import PPO

# --- IMPORTAÇÕES DO SISTEMA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from binance_connector import BinanceConnector
from manager import GerenciadorEstado

# Importa Features Engine
try:
    from features_engine import FeaturesEngine
except ImportError:
    sys.path.append(current_dir)
    from features_engine import FeaturesEngine

# --- CONFIGURAÇÃO ---
MODELO_NOME = "genesis_wld_v1"
MODELO_PATH = os.path.join(current_dir, "cerebros", MODELO_NOME)
PAR_ALVO = "WLDUSDT"
TIMEFRAME = "15m"
CAPITAL_OPERACIONAL = 200  
WINDOW_SIZE = 30 

def main():
    print(f"🤖 INICIANDO LIVE TRADER GÊNESIS ({PAR_ALVO})...")
    
    try:
        con = BinanceConnector()
        gerenciador = GerenciadorEstado()
        
        # 1. Carrega Modelo
        path = MODELO_PATH + ".zip" if not os.path.exists(MODELO_PATH) else MODELO_PATH
        if not os.path.exists(path):
            print(f"❌ Erro: Modelo não encontrado em {path}")
            return
        
        model = PPO.load(path)
        print("🧠 Cérebro Carregado!")
        
        # 2. BUSCA INTELIGENTE DO DATASET (CORREÇÃO DO CAMINHO)
        nome_arquivo = "dataset_wld_clean.csv"
        caminhos_possiveis = [
            os.path.join(parent_dir, nome_arquivo),            # ../dataset... (Caso padrão)
            os.path.join(parent_dir, "Binance", nome_arquivo), # ../Binance/dataset... (Caso irmão)
            os.path.join(current_dir, nome_arquivo),           # ./dataset... (Mesma pasta)
            nome_arquivo                                       # Raiz de execução
        ]
        
        df_ref_path = None
        for p in caminhos_possiveis:
            if os.path.exists(p):
                df_ref_path = p
                break
        
        if df_ref_path:
            print(f"📊 Dataset de Treino encontrado: {df_ref_path}")
            df_ref = pd.read_csv(df_ref_path)
            
            # Recalcula estatísticas do treino
            cols_ignore = ['timestamp', 'close', 'target']
            df_numeric = df_ref.select_dtypes(include=[np.number])
            df_features_ref = df_numeric.drop(columns=[c for c in cols_ignore if c in df_numeric.columns])
            
            global_mean = df_features_ref.mean()
            global_std = df_features_ref.std()
            
            # Guarda as colunas exatas para garantir a ordem
            COLS_TREINO = df_features_ref.columns.tolist()
            print(f"✅ Normalização Sincronizada ({len(COLS_TREINO)} features).")
        else:
            print(f"⚠️ AVISO CRÍTICO: '{nome_arquivo}' não encontrado em lugar nenhum!")
            print("   A IA vai usar normalização local (pode afetar a precisão).")
            global_mean, global_std, COLS_TREINO = None, None, None

    except Exception as e:
        print(f"❌ Falha na inicialização: {e}")
        import traceback
        traceback.print_exc()
        return

    em_posicao = False
    preco_entrada = 0
    lado_trade = None 
    
    print(f"🔭 Monitorando {PAR_ALVO}...")

    while True:
        try:
            time.sleep(2) 

            # 3. Coleta de Dados
            df_raw = con.buscar_candles(PAR_ALVO, TIMEFRAME, limit=200)
            df_btc = con.buscar_candles("BTCUSDT", TIMEFRAME, limit=200)
            if df_raw is None or df_btc is None: continue

            # 4. Engenharia de Features
            df_proc = FeaturesEngine.processar_dados(df_raw, df_btc)
            if df_proc is None or len(df_proc) < WINDOW_SIZE: continue

            # 5. Prepara Observação
            # Se temos a lista do treino, forçamos a mesma ordem/colunas
            if COLS_TREINO:
                for col in COLS_TREINO:
                    if col not in df_proc.columns: df_proc[col] = 0
                df_features = df_proc[COLS_TREINO].copy()
            else:
                # Fallback
                cols_ignore = ['timestamp', 'close', 'target']
                df_numeric = df_proc.select_dtypes(include=[np.number])
                df_features = df_numeric.drop(columns=[c for c in cols_ignore if c in df_numeric.columns])
            
            # Normalização (Aplica a régua do treino)
            if global_mean is not None:
                df_norm = (df_features - global_mean) / global_std
            else:
                df_norm = (df_features - df_features.mean()) / df_features.std()
                
            df_norm = df_norm.fillna(0).clip(-5, 5)
            
            # Janela Deslizante
            obs = df_norm.tail(WINDOW_SIZE).values.astype(np.float32)
            
            # Flatten (Shape fix)
            obs_flat = obs.flatten()
            
            # 6. Decisão
            action, _ = model.predict(obs_flat, deterministic=True)
            action = action.item()
            
            # Logs
            preco_atual = df_raw.iloc[-1]['close']
            sinal_str = "NEUTRO"
            cor = "\033[93m"
            if action == 1: sinal_str = "BUY"; cor = "\033[92m"
            elif action == 2: sinal_str = "SELL"; cor = "\033[91m"
            elif action == 3: sinal_str = "CLOSE"
            
            print(f"👀 {PAR_ALVO}: {cor}{sinal_str}{'\033[0m'} | Preço: {preco_atual}")

            # Atualiza Dashboard
            import pandas_ta as ta
            df_raw.ta.adx(length=14, append=True)
            adx_val = df_raw.iloc[-1].get('ADX_14', 0)
            
            status_dash = [{
                "par": PAR_ALVO, "preco": preco_atual, "adx": round(adx_val, 1),
                "sinal": sinal_str, "confianca": 100 if action != 0 else 0,
                "status_adx": "GENESIS-WLD"
            }]
            gerenciador.atualizar_monitor(status_dash)

            # 7. Execução
            if em_posicao:
                sair = False
                if action == 3: sair = True
                elif action == 1 and lado_trade == 2: sair = True
                elif action == 2 and lado_trade == 1: sair = True
                
                if sair:
                    print(f"🛡️ FECHANDO POSIÇÃO...")
                    lucro = (preco_atual - preco_entrada) / preco_entrada
                    if lado_trade == 2: lucro = -lucro
                    gerenciador.registrar_trade(PAR_ALVO, "CLOSE", preco_atual, 0, CAPITAL_OPERACIONAL, f"PnL: {lucro*100:.2f}%")
                    em_posicao = False

            if not em_posicao and action in [1, 2]:
                if gerenciador.pode_enviar_alerta(PAR_ALVO, TIMEFRAME):
                    tipo = "BUY" if action == 1 else "SELL"
                    print(f"\n🚀 GÊNESIS ORDENOU: {tipo}!")
                    
                    qtd = con.calcular_qtd_correta(PAR_ALVO, CAPITAL_OPERACIONAL, preco_atual)
                    if qtd > 0:
                        print(f"💰 Entrada: {preco_atual} | Qtd: {qtd}")
                        # con.colocar_ordem_market(...) 
                        em_posicao = True
                        lado_trade = action
                        preco_entrada = preco_atual
                        gerenciador.registrar_envio(PAR_ALVO)
                        gerenciador.registrar_trade(PAR_ALVO, tipo, preco_atual, qtd, CAPITAL_OPERACIONAL, "GENESIS-LIVE")

            time.sleep(10)

        except KeyboardInterrupt:
            print("\n🛑 Parando..."); break
        except Exception as e:
            print(f"❌ Erro Loop: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()