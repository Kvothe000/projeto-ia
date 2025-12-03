# Genesis_AI/fixed_trading_env.py (VERSÃO V2 - BONUS 1%)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class RealisticTradingEnv(gym.Env):
    def __init__(self, df_norm, df_price, initial_balance=10000, lookback_window=50):
        super(RealisticTradingEnv, self).__init__()
        
        # Dados
        self.df = df_norm
        
        # Tratamento Flexível de Preço
        if isinstance(df_price, pd.DataFrame) and 'close' in df_price.columns:
            self.price_data = df_price['close'].values
        elif hasattr(df_price, 'values'):
            self.price_data = df_price.values
        else:
            self.price_data = np.array(df_price)
            
        if len(self.price_data.shape) > 1: self.price_data = self.price_data.flatten()

        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # AÇÕES: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
        self.action_space = spaces.Discrete(4)
        
        # Observação
        self.obs_shape = lookback_window * self.df.shape[1]
        self.observation_space = spaces.Box(
            low=-10, high=10, 
            shape=(self.obs_shape,), 
            dtype=np.float32
        )
        
        self.taxa = 0.0005 # 0.05%
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        
        self.position = 0 
        self.entry_price = 0.0
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
        
        # PnL Flutuante
        unrealized_pnl = 0
        pct_change_trade = 0
        
        if self.position != 0:
            if self.position == 1: # Long
                pct_change_trade = (current_price - self.entry_price) / self.entry_price
            else: # Short
                pct_change_trade = (self.entry_price - current_price) / self.entry_price
            
            unrealized_pnl = self.position_vol_usd * pct_change_trade

        self.net_worth = self.balance + unrealized_pnl

        # --- LÓGICA DE EXECUÇÃO E RECOMPENSA ---
        
        # FECHAR (Close) ou INVERTER
        if self.position != 0 and ((action == 3) or (action == 1 and self.position == -1) or (action == 2 and self.position == 1)):
            # Realiza o Trade
            self.balance += unrealized_pnl
            custo = self.position_vol_usd * self.taxa
            self.balance -= custo
            
            # A recompensa base é o lucro em Dólar (normalizado para não explodir)
            # Ex: Ganhou $100 em $10000 = +1.0
            reward = (unrealized_pnl - custo) / self.initial_balance * 100
            
            # --- BÔNUS DA RIQUEZA (O SEGREDO DO 1%) ---
            # Se o lucro líquido for maior que 1% (0.01), dá um prêmio extra!
            if pct_change_trade >= 0.01:
                reward += 5.0 # Dopamina maciça para a IA priorizar alvos de 1%
                # print(f"🎯 BÔNUS 1% ATINGIDO! Lucro: {pct_change_trade*100:.2f}%")

            # Punição por prejuízo (Dor)
            if pct_change_trade < 0:
                reward *= 1.5 # A dor da perda é maior que a alegria do ganho (Psicologia)

            # Zera Posição
            self.position = 0
            self.position_vol_usd = 0
            self.entry_price = 0

        # ABRIR (Se zerado)
        if self.position == 0 and action in [1, 2]:
            self.position = 1 if action == 1 else -1
            self.entry_price = current_price
            # Aposta 100% do saldo (Juros Compostos Simulado no Treino)
            self.position_vol_usd = self.balance 
            self.balance -= self.position_vol_usd * self.taxa 
            self.net_worth = self.balance 

        # Punição leve por ficar exposto sem lucro (Time Decay)
        if self.position != 0 and pct_change_trade <= 0:
            reward -= 0.01

        # Fim do Jogo (Falência)
        terminated = False
        if self.net_worth <= self.initial_balance * 0.5: 
            terminated = True
            reward = -100 # Punição máxima

        return self._get_observation(), reward, terminated, False, {'net_worth': self.net_worth}