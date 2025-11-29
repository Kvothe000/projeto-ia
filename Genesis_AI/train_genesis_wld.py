# Genesis_AI/train_genesis_wld.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import os

# CONFIG
DADOS = "../Binance/dataset_wld_clean.csv"
MODELO = "cerebros/genesis_wld_v1"
WINDOW = 30

def main():
    print("🧬 TREINANDO GÊNESIS (ESPECIALISTA WLD)...")
    
    if not os.path.exists(DADOS):
        print("❌ Rode o gerar_dataset_wld.py primeiro!")
        return
        
    df = pd.read_csv(DADOS)
    
    # Prepara Dados
    price_data = df['close'].values
    
    # Features para a IA (Remove preço e timestamp)
    cols_drop = ['timestamp', 'close', 'target'] 
    df_obs = df.drop(columns=[c for c in cols_drop if c in df.columns])
    
    # Normaliza
    df_norm = (df_obs - df_obs.mean()) / df_obs.std()
    df_norm = df_norm.fillna(0).clip(-5, 5)
    
    # Ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(df_norm, price_data, lookback_window=WINDOW)])
    
    # Modelo
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048)
    
    print("🏋️ Treinando...")
    model.learn(total_timesteps=100000)
    
    model.save(MODELO)
    print("✅ Cérebro WLD Salvo!")

if __name__ == "__main__":
    main()