# Genesis_AI/fixed_trading_env.py (CORRIGIDO PELO COLEGA)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class RealisticTradingEnv(gym.Env):
    """
    AMBIENTE V13 - Lógica Financeira Corrigida + Observação Simplificada
    """
    def __init__(self, df_norm, df_price, initial_balance=10000, lookback_window=30):
        super(RealisticTradingEnv, self).__init__()
        
        # Dados
        self.df = df_norm.reset_index(drop=True)
        
        # Preço Real (Flexível)
        if isinstance(df_price, pd.DataFrame) and 'close' in df_price.columns:
            self.price_data = df_price['close'].values
        elif hasattr(df_price, 'values'):
            self.price_data = df_price.values
        else:
            self.price_data = np.array(df_price)
        if len(self.price_data.shape) > 1: self.price_data = self.price_data.flatten()

        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Ações: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # OBSERVAÇÃO: Vetor Achatado (Flattened)
        # Window * Features = Vetor único (Mais fácil para MLP)
        n_features = self.df.shape[1]
        self.obs_shape = lookback_window * n_features
        self.observation_space = spaces.Box(
            low=-10, high=10, 
            shape=(self.obs_shape,), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0 # 0=Flat, 1=Long, -1=Short
        self.position_size = 0.0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Pega janela de dados
        start = self.current_step - self.lookback_window
        end = self.current_step
        
        # Garante limites
        if start < 0: 
            obs = np.zeros((self.lookback_window, self.df.shape[1]))
        else:
            obs = self.df.iloc[start:end].values
            
        # Padding se faltar dados
        if len(obs) < self.lookback_window:
             padding = np.zeros((self.lookback_window - len(obs), self.df.shape[1]))
             obs = np.vstack([padding, obs])

        # ACHATA PARA VETOR (A grande mudança)
        return obs.flatten().astype(np.float32)

    def step(self, action):
        # 1. Avança Tempo
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {'net_worth': self.portfolio_value}

        # 2. Dados Atuais
        current_price = self.price_data[self.current_step]
        
        # 3. Executa Ação (Lógica Financeira Corrigida)
        # Se mudar de posição, fecha a anterior primeiro
        if self.position != 0 and ((action == 1 and self.position == -1) or (action == 2 and self.position == 1)):
            # Fecha posição atual
            if self.position == 1: # Fecha Long
                pnl = (current_price - self.entry_price) * self.position_size
            else: # Fecha Short
                pnl = (self.entry_price - current_price) * self.position_size
            
            self.balance += pnl
            self.position = 0
            self.position_size = 0

        # Abre Nova Posição (se não estiver posicionado ou tiver virado a mão)
        if action == 1 and self.position == 0: # BUY
            self.position = 1
            self.entry_price = current_price
            # Usa 95% do saldo para margem (simples)
            self.position_size = (self.balance * 0.95) / current_price
            
        elif action == 2 and self.position == 0: # SELL
            self.position = -1
            self.entry_price = current_price
            self.position_size = (self.balance * 0.95) / current_price

        # 4. Calcula Portfolio Value (CORREÇÃO MATEMÁTICA)
        floating_pnl = 0
        if self.position == 1:
            floating_pnl = (current_price - self.entry_price) * self.position_size
        elif self.position == -1:
            floating_pnl = (self.entry_price - current_price) * self.position_size
            
        self.portfolio_value = self.balance + floating_pnl
        
        # 5. Recompensa: Variação do Portfolio
        reward = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        self.prev_portfolio_value = self.portfolio_value
        
        # Escala a recompensa para a IA sentir mais
        reward *= 100 
        
        # 6. Kill Switch
        terminated = False
        if self.portfolio_value < self.initial_balance * 0.5:
            terminated = True
            reward = -10 # Punição final
            
        return self._get_observation(), reward, terminated, False, {'net_worth': self.portfolio_value}