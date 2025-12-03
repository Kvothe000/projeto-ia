# Genesis_AI/sleep_mode.py (CORRIGIDO: Mapeamento de Colunas + WLD)
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
# Aponta para o dataset da WLD que geramos
DATASET_TREINO_ORIGINAL = os.path.join(BASE_DIR, "..", "dataset_wld_clean.csv")
MEMORIA_RECENTE = os.path.join(BASE_DIR, "genesis_memory.csv")
# Aponta para o cérebro que estamos usando
MODELO_ATUAL = os.path.join(BASE_DIR, "cerebros", "genesis_wld_v2") # ou genesis_v12_final
WINDOW_SIZE = 30 

def ciclo_de_sono():
    print("🌙 INICIANDO CICLO DE SONO (WLD OPTIMIZED)...")
    
    # 1. Carrega Memórias do Dia
    if not os.path.exists(MEMORIA_RECENTE):
        print("💤 Nenhuma memória nova. A IA continua a dormir.")
        return

    try:
        df_recente = pd.read_csv(MEMORIA_RECENTE)
        
        # --- CORREÇÃO DE COMPATIBILIDADE ---
        # Se a memória antiga tiver 'real_price', renomeia para 'close'
        if 'real_price' in df_recente.columns:
            df_recente = df_recente.rename(columns={'real_price': 'close'})
            
        print(f"🧠 Processando {len(df_recente)} experiências do dia...")
    except Exception as e:
        print(f"❌ Erro ao ler memória: {e}"); return

    # 2. Carrega Histórico (Base de Conhecimento)
    df_treino_mix = df_recente
    
    if os.path.exists(DATASET_TREINO_ORIGINAL):
        df_antigo = pd.read_csv(DATASET_TREINO_ORIGINAL)
        
        # Garante que temos as mesmas colunas (Interseção)
        # 'close' é obrigatório, então forçamos sua presença
        cols_obrigatorias = list(set(df_antigo.columns) & set(df_recente.columns))
        
        if 'close' not in cols_obrigatorias:
            print("❌ Erro Crítico: Coluna 'close' perdida no cruzamento. Verifique os CSVs.")
            print(f"   Colunas Antigo: {list(df_antigo.columns)[:5]}...")
            print(f"   Colunas Novo: {list(df_recente.columns)[:5]}...")
            return

        # Mistura: 5000 candles antigos + O que aprendeu hoje
        # Isso impede o "Catastrophic Forgetting" (esquecer o passado)
        df_treino_mix = pd.concat([
            df_antigo[cols_obrigatorias].tail(5000), 
            df_recente[cols_obrigatorias]
        ], ignore_index=True)
        
        print(f"📚 Base Combinada: {len(df_treino_mix)} linhas (Passado + Presente).")
    else:
        print("⚠️ Aviso: Dataset original não encontrado. Treinando APENAS com memória recente (Risco alto).")

    # 3. Prepara Ambiente de Sonho
    try:
        # Separa Preço Real
        price_data = df_treino_mix['close'].values

        # Normaliza Features
        df_num = df_treino_mix.select_dtypes(include=[np.number])
        # Remove colunas que não são features de entrada da rede neural
        cols_drop = ['target', 'timestamp', 'close', 'action', 'reward', 'pnl_pct', 'real_price']
        df_features = df_num.drop(columns=[c for c in cols_drop if c in df_num.columns])
        
        # Normalização Z-Score
        df_norm = (df_features - df_features.mean()) / df_features.std()
        df_norm = df_norm.fillna(0).clip(-5, 5)
        
        # Cria Ambiente
        env = DummyVecEnv([lambda: RealisticTradingEnv(df_norm, price_data, lookback_window=WINDOW_SIZE)])
        
    except Exception as e:
        print(f"❌ Erro na preparação: {e}"); return

    # 4. Re-Treino (Fine-Tuning)
    print("🧘 Meditando sobre os trades...")
    try:
        # Carrega modelo atual
        path = MODELO_ATUAL + ".zip" if not os.path.exists(MODELO_ATUAL) else MODELO_ATUAL
        if not os.path.exists(path):
            print(f"❌ Modelo {path} não encontrado. Criando novo?")
            # Se não existir, poderíamos criar um novo, mas melhor avisar
            return

        model = PPO.load(path, env=env)
        
        # Treina pouco (Fine-tuning) para ajustar sem quebrar
        model.learn(total_timesteps=20000)
        
        # 5. Salvar Evolução
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        backup_path = f"{MODELO_ATUAL}_backup_{timestamp}"
        
        # Backup
        if os.path.exists(path):
            shutil.copy(path, backup_path + ".zip")
            print(f"📦 Backup criado: {backup_path}")
            
        # Salva Atualizado
        model.save(MODELO_ATUAL)
        
        print(f"✨ EVOLUÇÃO CONCLUÍDA! Cérebro atualizado.")
        
        # Limpa memória de curto prazo (renomeia para backup)
        novo_nome_memoria = MEMORIA_RECENTE + f".{timestamp}.bak"
        if os.path.exists(MEMORIA_RECENTE):
            os.rename(MEMORIA_RECENTE, novo_nome_memoria)
            print("🧹 Memória limpa para o novo dia.")

    except Exception as e:
        print(f"❌ Pesadelo (Erro no treino): {e}")

if __name__ == "__main__":
    ciclo_de_sono()