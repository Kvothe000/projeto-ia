# Genesis_AI/train_genesis_fixed.py
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os

class FixedCryptoTradingEnv(gym.Env):
    """AMBIENTE CORRIGIDO - Normalização e Recompensas Estáveis"""
    
    def __init__(self, df, lookback_window=30, initial_balance=10000):
        super(FixedCryptoTradingEnv, self).__init__()
        
        # Dados normalizados na entrada
        self.df = df
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        
        # Ações: 0=Hold, 1=Long, 2=Short, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Observação: Janela deslizante
        self.observation_space = spaces.Box(
            low=-10, high=10, 
            shape=(lookback_window, self.df.shape[1]), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_step = self.lookback_window
        self.position = 0 # 0=Flat, 1=Long, -1=Short
        self.entry_price = 0.0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        obs = self.df.iloc[self.current_step - self.lookback_window : self.current_step].values
        return obs.astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}
        
        # Preços Reais (Desnormalizados para cálculo de lucro)
        # Como normalizamos tudo, precisamos do preço real 'close'
        # Nota: Para simplificar, assumimos que a coluna 'close' é a 0
        # Mas como normalizamos, o valor é relativo. Vamos usar a variação percentual.
        
        # Variação Percentual Real (Estimada pela feature mom_1 se existir, ou close normalizado)
        # Vamos usar a variação do close normalizado como proxy
        curr_val = self.df.iloc[self.current_step, 0] # Close Norm
        prev_val = self.df.iloc[self.current_step-1, 0]
        # pct_change aproximado (não exato em dólares, mas direcionalmente correto para RL)
        pct_change = (curr_val - prev_val) * 0.01 # Escala reduzida
        
        reward = 0
        
        # Lógica Simplificada
        if action == 3 and self.position != 0: # Close
            self.position = 0
            reward -= 0.001 # Taxa
            
        elif action == 1: # Long
            if self.position == 0: self.position = 1
            elif self.position == -1: self.position = 1; reward -= 0.002
            
        elif action == 2: # Short
            if self.position == 0: self.position = -1
            elif self.position == 1: self.position = -1; reward -= 0.002
            
        # Recompensa
        if self.position == 1: reward += pct_change * 10
        elif self.position == -1: reward -= pct_change * 10
        
        return self._get_observation(), reward, False, False, {}

def main():
    print("🧬 INICIANDO TREINO GÊNESIS (CORRIGIDO)...")
    
    # 1. Carrega e Normaliza
    try:
        df = pd.read_csv('../Binance/dataset_v11_fusion.csv')
        df = df.select_dtypes(include=[np.number])
        
        print(f"📚 Normalizando {len(df)} linhas...")
        # Normalização Z-Score (Média 0, Desvio 1)
        df = (df - df.mean()) / df.std()
        df = df.fillna(0)
        # Clipa outliers extremos (evita explosão)
        df = df.clip(-5, 5)
        
    except Exception as e:
        print(f"❌ Erro dados: {e}")
        return

    # 2. Ambiente
    env = DummyVecEnv([lambda: FixedCryptoTradingEnv(df)])
    
    # 3. Agente (Configuração Conservadora)
    model = PPO(
        "MlpPolicy", # Começa com MLP (mais simples que LSTM)
        env,
        learning_rate=1e-4, # Lento e constante
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        clip_range=0.1, # Clip forte para estabilidade
        verbose=1,
        tensorboard_log="logs/genesis_fixed"
    )
    
    print("🎓 Iniciando Treino Estável...")
    model.learn(total_timesteps=200000)
    
    model.save("cerebros/genesis_v2_stable")
    print("✅ Treino Concluído!")

if __name__ == "__main__":
    main()