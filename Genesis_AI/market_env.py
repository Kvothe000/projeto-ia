# Genesis_AI/market_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CryptoGenesisEnv(gym.Env):
    """
    O Universo onde a IA Gênesis vive.
    Ela observa o mercado e toma decisões. O ambiente devolve:
    1. O novo estado do mercado.
    2. A recompensa (Dopamina) ou Punição (Dor).
    """
    def __init__(self, df, capital_inicial=1000.0):
        super(CryptoGenesisEnv, self).__init__()
        
        self.df = df
        self.capital_inicial = capital_inicial
        
        # AÇÕES POSSÍVEIS (O que ela pode fazer)
        # 0: NEUTRO (Ficar líquido/Observar)
        # 1: COMPRAR (Long)
        # 2: VENDER (Short)
        # 3: FECHAR (Zerar posição)
        self.action_space = spaces.Discrete(4)
        
        # PERCEPÇÃO (O que ela vê)
        # Vê todas as colunas do dataset (RSI, Volatilidade, Contexto BTC, etc)
        n_features = df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
        # Configurações de "Dor e Prazer"
        self.taxa_corretagem = 0.0005 # 0.05% (Taker)
        self.punicao_risco = 0.1      # Punição por volatilidade na carteira
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.capital_inicial
        self.net_worth = self.capital_inicial
        self.max_net_worth = self.capital_inicial
        self.current_step = 0
        
        # Estado da Posição
        self.position = 0      # 0=Flat, 1=Long, -1=Short
        self.entry_price = 0.0
        self.position_size = 0.0
        self.trades_history = []
        
        return self._next_observation(), {}

    def _next_observation(self):
        # A IA "olha" para o candle atual
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        
        # Dados de Mercado
        current_price = self.df.iloc[self.current_step]['close']
        prev_price = self.df.iloc[self.current_step - 1]['close']
        
        reward = 0
        terminated = False
        truncated = False
        
        # --- LÓGICA DE EXECUÇÃO ---
        
        # 3: FECHAR POSIÇÃO
        if action == 3 and self.position != 0:
            self.position = 0
            # Custo de transação (Punição leve)
            custo = current_price * self.position_size * self.taxa_corretagem
            self.balance -= custo
            self.net_worth -= custo
            reward -= 0.01 # Pequena dor para evitar overtrading
            
        # 1: LONG
        elif action == 1:
            if self.position == 0: # Entra
                self.position = 1
                self.entry_price = current_price
                # Define tamanho da mão (Fixa por enquanto, depois a IA decide)
                self.position_size = (self.balance * 0.99) / current_price
                # Paga taxa
                custo = self.balance * self.taxa_corretagem
                self.balance -= custo
                
            elif self.position == -1: # Vira a mão (Short -> Long)
                # Fecha Short
                self.position = 0
                # Abre Long
                self.position = 1
                self.entry_price = current_price
                reward -= 0.02 # Punição por indecisão (troca rápida)

        # 2: SHORT
        elif action == 2:
            if self.position == 0: # Entra
                self.position = -1
                self.entry_price = current_price
                self.position_size = (self.balance * 0.99) / current_price
                custo = self.balance * self.taxa_corretagem
                self.balance -= custo
                
            elif self.position == 1: # Vira a mão (Long -> Short)
                self.position = 0
                self.position = -1
                self.entry_price = current_price
                reward -= 0.02

        # --- CÁLCULO DO RESULTADO (AULA DE VIDA) ---
        
        step_return = 0
        if self.position == 1:
            step_return = (current_price - prev_price) / prev_price
        elif self.position == -1:
            step_return = (prev_price - current_price) / prev_price
            
        # Atualiza patrimônio
        self.net_worth *= (1 + step_return)
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # --- SISTEMA DE RECOMPENSA (A Alma do Negócio) ---
        
        # 1. Lucro é bom, mas Lucro consistente é melhor
        reward += step_return * 100 
        
        # 2. Punição por Drawdown (Perder dinheiro dói mais do que ganhar dá prazer)
        drawdown = (self.net_worth - self.max_net_worth) / self.max_net_worth
        if drawdown < -0.05: # Se perder 5% do topo
            reward -= 1.0 # Dor intensa
            
        # 3. Game Over (Quebrou a banca)
        if self.net_worth <= self.capital_inicial * 0.5:
            terminated = True
            reward = -1000 # Morte
            
        # Fim dos dados
        if self.current_step >= len(self.df) - 1:
            terminated = True
            reward += (self.net_worth - self.capital_inicial) * 0.1 # Bônus final se lucrou

        info = {
            'net_worth': self.net_worth,
            'drawdown': drawdown
        }
        
        return self._next_observation(), reward, terminated, truncated, info