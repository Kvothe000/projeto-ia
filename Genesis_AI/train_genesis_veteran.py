# Genesis_AI/train_genesis_veteran.py
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import os

# CONFIGURAÇÃO DE GUERRA
DADOS = "../Binance/dataset_wld_1ano.csv" # O dataset difícil
MODELO = "Genesis_AI/cerebros/genesis_wld_veteran"   # O novo cérebro experiente
WINDOW = 30

def main():
    print("🧬 INICIANDO TREINO VETERANO (1 ANO DE DADOS)...")
    
    if not os.path.exists(DADOS):
        print("❌ Erro: dataset_wld_1ano.csv não encontrado.")
        return
        
    df = pd.read_csv(DADOS)
    print(f"📚 Carregando {len(df)} candles de história completa...")
    
    # Prepara Dados
    price_data = df['close'].values
    cols_drop = ['timestamp', 'close', 'target'] 
    df_obs = df.drop(columns=[c for c in cols_drop if c in df.columns])
    
    # Normalização
    df_norm = (df_obs - df_obs.mean()) / df_obs.std()
    df_norm = df_norm.fillna(0).clip(-5, 5)
    
    # Ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(
        df_norm, 
        price_data, 
        initial_balance=10000, 
        lookback_window=WINDOW
    )])
    
    # Modelo PPO - Mais Paciente
    # Aumentamos n_epochs para ele "refletir" mais sobre cada erro difícil
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=4096,       # Memória de batch maior
        batch_size=256,     # Aprende em blocos maiores
        n_epochs=20,        # Estuda cada bloco 20 vezes
        gamma=0.995,        # Visão de longo prazo
        tensorboard_log="logs_veteran"
    )
    
    print("🏋️ Treinando o Veterano (Isso vai demorar)...")
    # 300.000 passos = Várias voltas no ano inteiro
    model.learn(total_timesteps=300000)
    
    model.save(MODELO)
    print(f"✅ CÉREBRO VETERANO SALVO: {MODELO}")
    print("👉 Teste-o com 'run_long_test.py' (altere o modelo no script para veteran).")

if __name__ == "__main__":
    main()