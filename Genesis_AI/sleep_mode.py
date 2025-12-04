# Genesis_AI/sleep_mode.py (VERSÃO: RECONCILIAÇÃO DE MEMÓRIA)
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv

# --- CONFIGURAÇÃO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_TREINO_ORIGINAL = os.path.join(BASE_DIR, "..", "dataset_wld_clean.csv")
ARQUIVO_FINANCEIRO = os.path.join(BASE_DIR, "..", "trades_history.csv")
MEMORIA_RECENTE = os.path.join(BASE_DIR, "genesis_memory.csv")
MODELO_ATUAL = os.path.join(BASE_DIR, "cerebros", "genesis_wld_v2") # Usamos a V2
WINDOW_SIZE = 30

def reconciliar_experiencias():
    """Cruza o que a IA viu com o que realmente aconteceu no bolso"""
    if not os.path.exists(MEMORIA_RECENTE) or not os.path.exists(ARQUIVO_FINANCEIRO):
        return None
    
    try:
        # Memória Sensorial (O que ela viu)
        df_mem = pd.read_csv(MEMORIA_RECENTE)
        # Memória Financeira (O resultado)
        df_fin = pd.read_csv(ARQUIVO_FINANCEIRO)
        
        # Converte timestamps para datetime para facilitar cruzamento (tolerância de 1 min)
        df_mem['dt'] = pd.to_datetime(df_mem['timestamp'], unit='s')
        df_fin['dt'] = pd.to_datetime(df_fin['data']) # Assume formato YYYY-MM-DD HH:MM:SS
        
        # Vamos atribuir recompensa aos passos que levaram ao trade
        # Simplificação: Se houve lucro no dia, reforça todas as decisões de compra
        # Lógica Pro: Matching exato é complexo. Vamos usar o PnL médio do dia como 'Bias'.
        
        pnl_acumulado = df_fin['pnl_pct'].sum()
        print(f"💰 Performance do Dia para Reconciliação: {pnl_acumulado:.2f}%")
        
        # Se o dia foi bom, aumenta o reward de todas as ações tomadas
        # Se foi ruim, penaliza.
        # Isso é um "Reforço de Tendência" simples mas eficaz para RL Online.
        if pnl_acumulado != 0:
            df_mem['reward'] = df_mem['reward'] + (pnl_acumulado * 0.1)
            
        return df_mem
        
    except Exception as e:
        print(f"⚠️ Erro na reconciliação: {e}")
        return pd.read_csv(MEMORIA_RECENTE) # Retorna memória bruta se falhar cruzamento

def ciclo_de_sono():
    print("🌙 INICIANDO CICLO DE SONO (AUTO-APRIMORAMENTO V2)...")
    
    # 1. Prepara Dados (Com Reconciliação)
    df_recente = reconciliar_experiencias()
    
    if df_recente is None or len(df_recente) < 50:
        print("💤 Poucas experiências vividas. A IA continua a dormir.")
        return

    print(f"🧠 Processando {len(df_recente)} memórias consolidadas...")

    # 2. Carrega Conhecimento Base (Para estabilidade)
    if os.path.exists(DATASET_TREINO_ORIGINAL):
        df_antigo = pd.read_csv(DATASET_TREINO_ORIGINAL)
        cols_comuns = [c for c in df_antigo.columns if c in df_recente.columns]
        
        # Mistura: 2000 candles antigos + O dia de hoje
        # Foco maior no recente para adaptação rápida
        df_treino_mix = pd.concat([df_antigo[cols_comuns].tail(2000), df_recente[cols_comuns]], ignore_index=True)
    else:
        df_treino_mix = df_recente

    # 3. Ambiente de Sonho
    try:
        # Garante coluna 'close'
        if 'close' not in df_treino_mix.columns and 'real_price' in df_treino_mix.columns:
            df_treino_mix = df_treino_mix.rename(columns={'real_price': 'close'})

        price_data = df_treino_mix['close'].values

        # Features e Normalização
        df_num = df_treino_mix.select_dtypes(include=[np.number])
        cols_drop = ['target', 'timestamp', 'close', 'action', 'reward', 'pnl_pct', 'dt']
        df_features = df_num.drop(columns=[c for c in cols_drop if c in df_num.columns])
        
        df_norm = (df_features - df_features.mean()) / df_features.std()
        df_norm = df_norm.fillna(0).clip(-5, 5)
        
        env = DummyVecEnv([lambda: RealisticTradingEnv(df_norm, price_data, lookback_window=WINDOW_SIZE)])
        
    except Exception as e:
        print(f"❌ Erro preparação: {e}"); return

    # 4. Re-Treino
    print("🧘 Meditando sobre os lucros e perdas...")
    try:
        path = MODELO_ATUAL + ".zip" if not os.path.exists(MODELO_ATUAL) else MODELO_ATUAL
        model = PPO.load(path, env=env)
        
        # Treino Rápido de Adaptação
        model.learn(total_timesteps=30000)
        
        # 5. Salvar
        timestamp = datetime.now().strftime("%Y%m%d")
        shutil.copy(path, f"{MODELO_ATUAL}_backup_{timestamp}.zip")
        model.save(MODELO_ATUAL)
        
        print(f"✨ EVOLUÇÃO CONCLUÍDA! Cérebro atualizado com sucesso.")
        
        # Limpeza
        if os.path.exists(MEMORIA_RECENTE):
            os.remove(MEMORIA_RECENTE) # Apaga memória curta pois já foi absorvida
            # Cria arquivo vazio para amanhã
            pd.DataFrame(columns=df_recente.columns).to_csv(MEMORIA_RECENTE, index=False)
            print("🧹 Mente limpa para o novo dia.")

    except Exception as e:
        print(f"❌ Pesadelo: {e}")

if __name__ == "__main__":
    ciclo_de_sono()