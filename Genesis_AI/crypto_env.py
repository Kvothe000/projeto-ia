# Genesis_AI/crypto_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CryptoTradingEnv(gym.Env):
    """
    A Arena onde a IA vai aprender a viver.
    Observação: O Mercado (Preço, Volume, Contexto BTC, Sentimento).
    Ação: Comprar, Vender, Segurar, Fechar.
    Recompensa: Lucro Ajustado ao Risco (Sharpe Ratio).
    """
    def __init__(self, df, capital_inicial=10000):
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df
        self.capital_inicial = capital_inicial
        
        # DEFINIÇÃO DE AÇÃO (O que a IA pode fazer)
        # 0: Hold/Neutro
        # 1: Buy (Long)
        # 2: Sell (Short)
        # 3: Close Position (Zerar)
        self.action_space = spaces.Discrete(4)
        
        # DEFINIÇÃO DE OBSERVAÇÃO (O que a IA vê)
        # Ela vê todas as colunas numéricas do nosso Dataset V11
        # (RSI, Volatilidade, Contexto BTC, etc.)
        n_features = df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
        # Estado Interno
        self.reset()

    def reset(self, seed=None):
        self.balance = self.capital_inicial
        self.net_worth = self.capital_inicial
        self.current_step = 0
        self.position = 0 # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0
        self.history = []
        
        return self._next_observation(), {}

    def _next_observation(self):
        # A IA vê a linha atual do CSV (O "Agora")
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs

    def step(self, action):
        # Avança o tempo
        self.current_step += 1
        
        # Dados atuais e anteriores
        current_price = self.df.iloc[self.current_step]['close']
        prev_price = self.df.iloc[self.current_step - 1]['close']
        
        reward = 0
        done = False
        
        # --- LÓGICA DE EXECUÇÃO DA IA ---
        
        # Se decidir FECHAR (3)
        if action == 3 and self.position != 0:
            self.position = 0
            # Pequena punição por operar demais (custo de taxa)
            reward -= 0.001 
            
        # Se decidir COMPRAR (1)
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1: # Vira a mão (Era short, virou long)
                self.position = 1
                self.entry_price = current_price
                
        # Se decidir VENDER (2)
        elif action == 2:
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1: # Vira a mão
                self.position = -1
                self.entry_price = current_price

        # --- CÁLCULO DA RECOMPENSA (DOPAMINA) ---
        # A recompensa é a variação do patrimônio a cada passo
        # Se Long e subiu: Recompensa positiva
        # Se Short e caiu: Recompensa positiva
        
        step_return = 0
        if self.position == 1:
            step_return = (current_price - prev_price) / prev_price
        elif self.position == -1:
            step_return = (prev_price - current_price) / prev_price
            
        # Atualiza patrimônio
        self.net_worth *= (1 + step_return)
        
        # A recompensa é o lucro, mas punimos volatilidade excessiva (Risco)
        reward += step_return * 100 
        
        # Punição por falência
        if self.net_worth <= self.capital_inicial * 0.5:
            done = True
            reward = -1000 # Dor extrema se perder metade do dinheiro
            
        # Fim dos dados
        if self.current_step >= len(self.df) - 1:
            done = True
            
        info = {'net_worth': self.net_worth}
        return self._next_observation(), reward, done, False, info