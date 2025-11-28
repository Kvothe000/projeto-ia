# Genesis_AI/crypto_env_advanced.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

class AdvancedCryptoTradingEnv(gym.Env):
    """
    A Arena de Treino Gênesis.
    Aqui a IA aprende a gerir risco, capital e emoções (drawdown).
    """
    def __init__(self, df, capital_inicial=10000, lookback_window=50):
        super(AdvancedCryptoTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.capital_inicial = capital_inicial
        self.lookback_window = lookback_window
        
        # AÇÕES: [0: Hold, 1: Buy, 2: Sell, 3: Close]
        # Simplificado para começar, depois expandimos para tamanho de posição
        self.action_space = spaces.Discrete(4)
        
        # OBSERVAÇÃO: Janela de tempo (Olhar para o passado) + Features Atuais
        n_features = df.shape[1]
        # Box(low, high, shape) -> Matriz de (50 candles x N features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window, n_features), 
            dtype=np.float32
        )
        
        self.taxa = 0.0005 # 0.05% por trade
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.capital_inicial
        self.net_worth = self.capital_inicial
        self.max_net_worth = self.capital_inicial
        
        # Começa num ponto aleatório para não decorar o início do dataset
        self.current_step = self.lookback_window 
        
        self.position = 0 
        self.entry_price = 0.0
        self.trades = []
        self.consecutive_losses = 0
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Retorna os últimos N candles (Memória de Curto Prazo)
        frame = self.df.iloc[self.current_step - self.lookback_window : self.current_step]
        return frame.values.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Dados atuais
        current_price = self.df.iloc[self.current_step]['close']
        prev_price = self.df.iloc[self.current_step - 1]['close']
        
        reward = 0
        terminated = False
        truncated = False
        
        # --- EXECUÇÃO ---
        
        # 3: CLOSE
        if action == 3 and self.position != 0:
            # Calcula Lucro Real
            pnl = 0
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price
            elif self.position == -1:
                pnl = (self.entry_price - current_price) / self.entry_price
            
            # Aplica na conta
            lucro_usd = self.balance * pnl
            custo = self.balance * self.taxa
            self.balance += (lucro_usd - custo)
            self.net_worth = self.balance
            
            # Recompensa/Punição
            if pnl > 0:
                reward += pnl * 100 # Dopamina pelo lucro
                self.consecutive_losses = 0
            else:
                reward += pnl * 150 # Dor extra pela perda (Aversão ao Risco)
                self.consecutive_losses += 1
            
            self.position = 0

        # 1: BUY
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.balance * self.taxa # Taxa entrada
            elif self.position == -1: # Vira mão
                # Fecha short (Custo duplo)
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.balance * (self.taxa * 2)
                reward -= 0.1 # Punição por indecisão

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
                reward -= 0.1

        # --- RECOMPENSA CONTÍNUA (SHAPING) ---
        
        # Variação do patrimônio (mesmo não realizado)
        step_return = 0
        if self.position == 1:
            step_return = (current_price - prev_price) / prev_price
        elif self.position == -1:
            step_return = (prev_price - current_price) / prev_price
            
        self.net_worth *= (1 + step_return)
        reward += step_return * 10 # Recompensa imediata por estar no lado certo

        # Punição por Drawdown (Medo)
        drawdown = (self.net_worth - self.max_net_worth) / self.max_net_worth
        if drawdown < -0.10: # Perdeu 10% do topo
            reward -= 2.0 # Pânico
            
        # Game Over
        if self.net_worth <= self.capital_inicial * 0.5:
            terminated = True
            reward = -1000
            
        # Fim do Dataset
        if self.current_step >= len(self.df) - 1:
            terminated = True

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        info = {'net_worth': self.net_worth}
        return self._next_observation(), reward, terminated, truncated, info