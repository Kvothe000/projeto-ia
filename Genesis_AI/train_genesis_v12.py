# Genesis_AI/train_genesis_v12.py (O TREINADOR BLINDADO)
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import os

DADOS_PATH = "../Binance/dataset_v11_fusion.csv"
MODELO_PATH = "cerebros/genesis_v12_final"
LOG_DIR = "logs_v12"

def main():
    print("🧬 INICIANDO TREINO V12 (CORREÇÃO DE ESCALA)...")
    
    # 1. Carrega Dados Brutos
    try:
        df_bruto = pd.read_csv(DADOS_PATH)
        df_bruto = df_bruto.select_dtypes(include=[np.number])
        
        # 2. SEPARA PREÇO REAL
        # O ambiente precisa do preço real (close) para cálculo de lucro
        df_price_real = df_bruto[['close']].copy()

        # 3. NORMALIZA DADOS (Z-Score)
        df_norm = df_bruto.copy()
        
        # Remove a coluna 'close' antes da normalização e usa o preço real
        if 'close' in df_norm.columns:
            df_norm = df_norm.drop(columns=['close'])
        
        # Aplica Normalização Z-Score
        df_norm = (df_norm - df_norm.mean()) / df_norm.std()
        df_norm = df_norm.fillna(0).clip(-5, 5)
        
    except Exception as e:
        print(f"❌ Erro carregando ou normalizando dados: {e}")
        return
    
    # 4. Cria Ambiente (Passando dados normalizados e o preço real)
    env = DummyVecEnv([lambda: RealisticTradingEnv(df_norm, df_price_real)])
    
    # 5. Agente PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,  # Learning rate baixo para segurança
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        clip_range=0.1,
        verbose=1,
        tensorboard_log=LOG_DIR
    )
    
    print("🎯 Iniciando Treino Estável (200k steps)...")
    model.learn(total_timesteps=200000)
    
    model.save(MODELO_PATH)
    print("✅ TREINO CONCLUÍDO! Cérebro Gênesis V12 salvo.")

if __name__ == "__main__":
    main()