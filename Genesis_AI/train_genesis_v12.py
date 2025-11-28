# Genesis_AI/train_genesis_v12.py (VERSÃO CORRIGIDA)
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import os

DADOS_PATH = "../Binance/dataset_v11_fusion.csv"
MODELO_PATH = "cerebros/genesis_v12_final"
LOG_DIR = "logs_v12"
WINDOW_SIZE = 50

def main():
    print("🧬 INICIANDO TREINO V12 (CORRIGIDO - SEM DATA LEAKAGE)...")
    
    try:
        df_bruto = pd.read_csv(DADOS_PATH)
        
        # 1. SEPARA TREINO/TESTE PARA EVITAR DATA LEAKAGE
        train_size = int(0.8 * len(df_bruto))
        df_train = df_bruto.iloc[:train_size].copy()
        df_test = df_bruto.iloc[train_size:].copy()
        
        print(f"📊 Dados: {len(df_train)} treino, {len(df_test)} teste")
        
        # 2. PREPARA FEATURES (apenas treino para estatísticas)
        df_price_real_train = df_train[['close']].copy()
        df_features_train = df_train.select_dtypes(include=[np.number]).copy()
        
        # Remove colunas proibidas
        cols_to_drop = ['target', 'timestamp', 'close']
        df_features_train = df_features_train.drop(
            columns=[c for c in cols_to_drop if c in df_features_train.columns]
        )
        
        print(f"🔧 Features ({len(df_features_train.columns)}): {list(df_features_train.columns)}")
        
        # 3. NORMALIZAÇÃO APENAS COM DADOS DE TREINO
        train_mean = df_features_train.mean()
        train_std = df_features_train.std()
        
        df_norm_train = (df_features_train - train_mean) / train_std
        df_norm_train = df_norm_train.fillna(0).clip(-8, 8)  # Clip menos agressivo
        
        # 4. PREPARA DADOS DE TESTE (com mesma normalização do treino)
        df_price_real_test = df_test[['close']].copy()
        df_features_test = df_test.select_dtypes(include=[np.number]).copy()
        df_features_test = df_features_test.drop(
            columns=[c for c in cols_to_drop if c in df_features_test.columns]
        )
        df_norm_test = (df_features_test - train_mean) / train_std
        df_norm_test = df_norm_test.fillna(0).clip(-8, 8)
        
    except Exception as e:
        print(f"❌ Erro preparação: {e}")
        return
    
    # 5. CRIA AMBIENTE DE TREINO
    env = DummyVecEnv([lambda: RealisticTradingEnv(
        df_norm_train, 
        df_price_real_train, 
        lookback_window=WINDOW_SIZE
    )])
    
    # 6. AGENTE COM HIPERPARÂMETROS MELHORADOS
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,                    # Adicionado
        gamma=0.99,
        gae_lambda=0.95,                # Adicionado
        clip_range=0.2,                 # Aumentado
        clip_range_vf=0.2,              # Adicionado
        normalize_advantage=True,       # Adicionado
        ent_coef=0.01,                  # Adicionado
        vf_coef=0.5,                    # Adicionado
        max_grad_norm=0.5,              # Adicionado (CRÍTICO!)
        tensorboard_log=LOG_DIR,
        verbose=1,
        device="auto"
    )
    
    print("🎯 Iniciando Treino (200k steps)...")
    
    # 7. TREINO COM CALLBACKS PARA MONITORAR
    model.learn(
        total_timesteps=200000,
        tb_log_name="genesis_v12",
        reset_num_timesteps=True
    )
    
    model.save(MODELO_PATH)
    
    # 8. TESTE RÁPIDO NO CONJUNTO DE TESTE
    print("🧪 Testando no conjunto de teste...")
    env_test = DummyVecEnv([lambda: RealisticTradingEnv(
        df_norm_test,
        df_price_real_test, 
        lookback_window=WINDOW_SIZE
    )])
    
    obs = env_test.reset()
    total_reward = 0
    for i in range(1000):  # Testa 1000 steps
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env_test.step(action)
        total_reward += rewards[0]
        if dones[0]:
            break
    
    print(f"📈 Recompensa no teste: {total_reward:.2f}")
    env_test.close()
    
    print("✅ TREINO CONCLUÍDO! Cérebro Gênesis V12 Corrigido salvo.")

if __name__ == "__main__":
    main()