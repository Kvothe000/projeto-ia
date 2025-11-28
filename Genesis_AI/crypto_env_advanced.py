# Genesis_AI/crypto_env_advanced.py (VERSÃO SEGURA)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AdvancedCryptoTradingEnv(gym.Env):
    def __init__(self, df, capital_inicial=10000, lookback_window=50):
        super(AdvancedCryptoTradingEnv, self).__init__()
        
        # GARANTIA DE DADOS
        self.df = df.reset_index(drop=True)
        
        # Verifica se a coluna 'close' existe (Crucial!)
        if 'close' not in self.df.columns:
            print(f"❌ ERRO CRÍTICO: Coluna 'close' não encontrada no Dataset!")
            print(f"   Colunas disponíveis: {list(self.df.columns)}")
            raise ValueError("Dataset inválido para o Ambiente.")

        self.capital_inicial = capital_inicial
        self.lookback_window = lookback_window
        
        # AÇÕES: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # OBSERVAÇÃO
        n_features = df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window, n_features), 
            dtype=np.float32
        )
        
        self.taxa = 0.0005
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.capital_inicial
        self.net_worth = self.capital_inicial
        self.max_net_worth = self.capital_inicial
        
        # Início seguro (longe do fim e do começo)
        self.current_step = self.lookback_window
        
        self.position = 0 
        self.entry_price = 0.0
        self.consecutive_losses = 0
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Garante que slice retorna numpy array float32
        obs = self.df.iloc[self.current_step - self.lookback_window : self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        # Avança
        self.current_step += 1
        
        # Verifica fim dos dados
        if self.current_step >= len(self.df):
            self.current_step = self.lookback_window # Loop ou Terminate
            return self._next_observation(), 0, True, False, {'net_worth': self.net_worth}

        # Dados Atuais
        current_price = self.df.iloc[self.current_step]['close']
        prev_price = self.df.iloc[self.current_step - 1]['close']
        
        reward = 0
        terminated = False
        
        # --- EXECUÇÃO (Mesma lógica anterior) ---
        # 3: CLOSE
        if action == 3 and self.position != 0:
            pnl = 0
            if self.position == 1: pnl = (current_price - self.entry_price) / self.entry_price
            elif self.position == -1: pnl = (self.entry_price - current_price) / self.entry_price
            
            self.balance += self.balance * pnl - (self.balance * self.taxa)
            self.net_worth = self.balance
            
            if pnl > 0: reward += pnl * 100
            else: reward += pnl * 150
            
            self.position = 0

        # 1: BUY
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.balance * self.taxa
            elif self.position == -1:
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.balance * (self.taxa * 2)

        # 2: SELL
        elif action == 2:
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.balance * self.taxa
            elif self.position == 1:
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.balance * (self.taxa * 2)

        # --- SHAPING DE RECOMPENSA ---
        step_return = 0
        if self.position == 1: step_return = (current_price - prev_price) / prev_price
        elif self.position == -1: step_return = (prev_price - current_price) / prev_price
            
        self.net_worth *= (1 + step_return)
        reward += step_return * 10
        
        # Punição Drawdown
        if self.net_worth > self.max_net_worth: self.max_net_worth = self.net_worth
        drawdown = (self.net_worth - self.max_net_worth) / self.max_net_worth
        if drawdown < -0.10: reward -= 1.0

        # Game Over
        if self.net_worth <= self.capital_inicial * 0.5:
            terminated = True
            reward = -1000

        return self._next_observation(), reward, terminated, False, {'net_worth': self.net_worth}