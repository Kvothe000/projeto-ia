# Genesis_AI/quick_test.py
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np
import os

class SimpleTestEnv(gym.Env):
    """Ambiente SUPER SIMPLES para validar o treino"""
    
    def __init__(self, df):
        super().__init__()
        self.df = df.values
        # Ações: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        # Observação: 10 features do momento
        self.observation_space = spaces.Box(low=-5, high=5, shape=(10,), dtype=np.float32)
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # Retorna a primeira linha (primeiras 10 colunas)
        return self.df[0].astype(np.float32), {}
    
    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 100
        
        # Recompensa Fictícia para Teste:
        # Se 'mom_5' (coluna 1) for positiva e a ação for 1 (Buy) -> Ganha
        momentum = self.df[min(self.step_count, len(self.df)-1)][1]
        
        reward = 0
        if action == 1 and momentum > 0: reward = 1
        elif action == 2 and momentum < 0: reward = 1
        elif action == 0: reward = -0.1 # Punição por inércia
        
        obs = self.df[min(self.step_count, len(self.df)-1)].astype(np.float32)
        return obs, reward, done, False, {}

def quick_test():
    print("🚀 TESTE RÁPIDO GENESIS (Validando Ambiente)...")
    
    if not os.path.exists("../Binance/dataset_v11_fusion.csv"):
        print("❌ Dataset não encontrado na pasta Binance!")
        return

    # Carrega e Normaliza
    df = pd.read_csv('../Binance/dataset_v11_fusion.csv')
    df = df.select_dtypes(include=[np.number])
    df = (df - df.mean()) / df.std() # Normalização Z-Score (Crucial!)
    df = df.fillna(0)
    
    print(f"📚 Dados Normalizados: {len(df)} linhas")
    
    # Usa apenas as primeiras 10 colunas para o teste ser leve
    env = SimpleTestEnv(df.iloc[:, :10])
    
    # Modelo Leve (MLP)
    model = PPO("MlpPolicy", env, learning_rate=1e-3, verbose=1)
    
    print("🏋️ Treinando por 5000 passos...")
    model.learn(total_timesteps=5000)
    
    print("✅ Teste rápido concluído! Se o 'ep_rew_mean' subiu, estamos vivos.")

if __name__ == "__main__":
    quick_test()