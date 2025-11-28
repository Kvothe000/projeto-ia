# Genesis_AI/fixed_trading_env.py (VERSÃO FLEXÍVEL)
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class RealisticTradingEnv(gym.Env):
    def __init__(self, df_norm, df_price, initial_balance=10000, lookback_window=50):
        super(RealisticTradingEnv, self).__init__()
        
        # Dados Normalizados (O que a IA vê)
        self.df = df_norm
        
        # Preço REAL (Para calcular lucro) - CORREÇÃO DE FLEXIBILIDADE
        if isinstance(df_price, pd.DataFrame) and 'close' in df_price.columns:
            self.price_data = df_price['close'].values
        elif hasattr(df_price, 'values'):
             # Se for Series ou DataFrame sem coluna 'close' explícita
            self.price_data = df_price.values
        else:
            self.price_data = np.array(df_price)
            
        # Garante array 1D (Lista simples)
        if len(self.price_data.shape) > 1:
            self.price_data = self.price_data.flatten()

        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Ações
        self.action_space = spaces.Discrete(4)
        
        # Espaço de observação
        self.observation_space = spaces.Box(
            low=-5, high=5, 
            shape=(lookback_window, self.df.shape[1]), 
            dtype=np.float32
        )
        
        self.taxa = 0.0005
        self.reset()
    
    def _safe_normalize(self, df):
        df_normalized = df.copy()
        df_normalized = df_normalized.select_dtypes(include=[np.number])
        
        for col in df_normalized.columns:
            if df_normalized[col].std() > 0:
                df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()
                df_normalized[col] = np.clip(df_normalized[col], -5, 5)
            else:
                df_normalized[col] = 0.0
        
        return df_normalized.fillna(0)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_step = self.lookback_window
        self.position = 0 
        self.entry_price = 0.0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        start = self.current_step - self.lookback_window
        obs = self.df.iloc[start:self.current_step].values
        
        if len(obs) < self.lookback_window:
             padding = np.zeros((self.lookback_window - len(obs), self.df.shape[1]))
             obs = np.vstack([padding, obs])

        return obs.astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {'net_worth': self.net_worth}
        
        # Preços Reais
        current_price = self.price_data[self.current_step]
        
        # Lógica de Posição
        reward = 0
        
        # 3: Close
        if action == 3 and self.position != 0: 
            pnl = 0
            if self.position == 1: pnl = (current_price - self.entry_price) / self.entry_price
            elif self.position == -1: pnl = (self.entry_price - current_price) / self.entry_price
            
            self.net_worth += (self.net_worth * pnl) - (self.net_worth * self.taxa)
            self.position = 0
            
        # 1: Long
        elif action == 1: 
            if self.position == 0: 
                self.position = 1
                self.entry_price = current_price
                self.net_worth -= self.net_worth * self.taxa
            elif self.position == -1: 
                self.position = 1
                self.entry_price = current_price
                self.net_worth -= self.net_worth * (self.taxa * 2)

        # 2: Short
        elif action == 2: 
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                self.net_worth -= self.net_worth * self.taxa
            elif self.position == 1:
                self.position = -1
                self.entry_price = current_price
                self.net_worth -= self.net_worth * (self.taxa * 2)

        # Recompensa Contínua (Variação Patrimonial)
        # Usamos a variação percentual do preço real para ajustar o net_worth simulado
        prev_price = self.price_data[self.current_step - 1]
        pct_change = (current_price - prev_price) / prev_price
        
        if self.position == 1: 
            self.net_worth *= (1 + pct_change)
        elif self.position == -1: 
            self.net_worth *= (1 - pct_change)
            
        # Falência
        terminated = False
        if self.net_worth <= self.initial_balance * 0.1:
            terminated = True
            
        return self._get_observation(), reward, terminated, False, {'net_worth': self.net_worth}