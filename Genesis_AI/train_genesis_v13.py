# Genesis_AI/train_genesis_v13.py
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import os

DADOS_PATH = "../Binance/dataset_v11_fusion.csv"
MODELO_PATH = "cerebros/genesis_v13_corrected"
LOG_DIR = "logs_v13"
WINDOW_SIZE = 30 # Reduzido para simplificar

def main():
    print("🧬 INICIANDO TREINO V13 (MATEMÁTICA CORRIGIDA)...")
    
    try:
        df_bruto = pd.read_csv(DADOS_PATH)
        
        # Separa Preço
        df_price = df_bruto[['close']].copy()
        
        # Prepara Features (Normalizadas)
        df_feat = df_bruto.select_dtypes(include=[np.number])
        cols_drop = ['target', 'timestamp', 'close']
        df_feat = df_feat.drop(columns=[c for c in cols_drop if c in df_feat.columns])
        
        df_norm = (df_feat - df_feat.mean()) / df_feat.std()
        df_norm = df_norm.fillna(0).clip(-5, 5)
        
        print(f"📚 Dados: {len(df_norm)} linhas. Features: {df_norm.shape[1]}")
        
    except Exception as e:
        print(f"❌ Erro dados: {e}")
        return

    # Cria Ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(df_norm, df_price, lookback_window=WINDOW_SIZE)])
    
    # Modelo PPO (Simples e Eficiente)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log=LOG_DIR
    )
    
    print("🎯 Treinando (100k steps)...")
    model.learn(total_timesteps=100000)
    
    model.save(MODELO_PATH)
    print(f"✅ MODELO SALVO: {MODELO_PATH}")

if __name__ == "__main__":
    main()