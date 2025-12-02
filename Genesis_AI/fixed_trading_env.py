# Genesis_AI/fixed_trading_env.py (CORREÇÃO MATEMÁTICA PnL)
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class RealisticTradingEnv(gym.Env):
    def __init__(self, df_norm, df_price, initial_balance=10000, lookback_window=50):
        super(RealisticTradingEnv, self).__init__()
        
        self.df = df_norm
        
        # Tratamento de Preço
        if isinstance(df_price, pd.DataFrame) and 'close' in df_price.columns:
            self.price_data = df_price['close'].values
        elif hasattr(df_price, 'values'):
            self.price_data = df_price.values
        else:
            self.price_data = np.array(df_price)
            
        if len(self.price_data.shape) > 1: self.price_data = self.price_data.flatten()

        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.action_space = spaces.Discrete(4)
        
        # Observação
        self.obs_shape = lookback_window * self.df.shape[1]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.obs_shape,), dtype=np.float32)
        
        self.taxa = 0.0005
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        
        # Saldo Disponível (Cash)
        self.balance = self.initial_balance
        # Patrimônio Total (Cash + Posição Aberta)
        self.net_worth = self.initial_balance
        
        self.position = 0 
        self.entry_price = 0.0
        # Tamanho da posição em Dólares (fixo na entrada)
        self.position_vol_usd = 0.0 
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        start = self.current_step - self.lookback_window
        obs = self.df.iloc[start:self.current_step].values
        if len(obs) < self.lookback_window:
             padding = np.zeros((self.lookback_window - len(obs), self.df.shape[1]))
             obs = np.vstack([padding, obs])
        return obs.flatten().astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {'net_worth': self.net_worth}
        
        current_price = self.price_data[self.current_step]
        reward = 0
        
        # --- 1. CÁLCULO DO PnL FLUTUANTE (CORREÇÃO) ---
        unrealized_pnl = 0
        if self.position != 0:
            if self.position == 1: # Long
                pct_change = (current_price - self.entry_price) / self.entry_price
            else: # Short
                pct_change = (self.entry_price - current_price) / self.entry_price
            
            # Lucro = Valor Apostado * % Variação
            unrealized_pnl = self.position_vol_usd * pct_change

        # Atualiza Patrimônio Total (Sem juros compostos)
        self.net_worth = self.balance + unrealized_pnl

        # --- 2. EXECUÇÃO DE AÇÕES ---
        
        # FECHAR (Close) ou INVERTER
        if self.position != 0 and ((action == 3) or (action == 1 and self.position == -1) or (action == 2 and self.position == 1)):
            # Realiza o Lucro/Prejuízo
            self.balance += unrealized_pnl
            self.balance -= self.position_vol_usd * self.taxa # Taxa Saída
            
            reward = unrealized_pnl # A recompensa é o dinheiro ganho
            
            # Zera
            self.position = 0
            self.position_vol_usd = 0
            self.entry_price = 0
            unrealized_pnl = 0

        # ABRIR (Se estiver zerado)
        if self.position == 0 and action in [1, 2]:
            self.position = 1 if action == 1 else -1
            self.entry_price = current_price
            
            # Aposta 100% do saldo atual (Simulação de All-In Composto por Trade, não por candle)
            # Ou use um valor fixo para ser mais conservador
            self.position_vol_usd = self.balance 
            
            self.balance -= self.position_vol_usd * self.taxa # Taxa Entrada
            
            # Recalcula Net Worth inicial pós-taxa
            self.net_worth = self.balance 

        # --- 3. PROTEÇÃO E FIM ---
        terminated = False
        if self.net_worth <= self.initial_balance * 0.5: # Perdeu 50%
            terminated = True
            reward = -100 # Punição

        return self._get_observation(), reward, terminated, False, {'net_worth': self.net_worth}