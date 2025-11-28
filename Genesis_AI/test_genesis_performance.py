# Genesis_AI/test_genesis_performance.py
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os

# --- DEFINIÇÃO DO AMBIENTE (Cópia exata do Treino para compatibilidade) ---
class FixedCryptoTradingEnv(gym.Env):
    def __init__(self, df, lookback_window=30, initial_balance=10000):
        super(FixedCryptoTradingEnv, self).__init__()
        self.df = df
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(4)
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
        self.position = 0
        self.entry_price = 0.0
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.iloc[self.current_step - self.lookback_window : self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {'net_worth': self.net_worth}
        
        # Preços simulados (Normalized Close como proxy de variação)
        curr_val = self.df.iloc[self.current_step, 0]
        prev_val = self.df.iloc[self.current_step-1, 0]
        pct_change = (curr_val - prev_val) * 0.01 # Escala reduzida
        
        reward = 0
        if action == 3 and self.position != 0: self.position = 0
        elif action == 1: self.position = 1
        elif action == 2: self.position = -1
            
        if self.position == 1: reward += pct_change * 10
        elif self.position == -1: reward -= pct_change * 10
        
        # Atualiza Net Worth (Simulado)
        self.net_worth *= (1 + (pct_change if self.position == 1 else -pct_change if self.position == -1 else 0))
        
        return self._get_observation(), reward, False, False, {'net_worth': self.net_worth}

# --- CLASSE DE TESTE ---
class PerformanceTester:
    def __init__(self, model_path, test_data_path):
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
            
        self.model = PPO.load(model_path)
        self.raw_df = pd.read_csv(test_data_path)
        
        # PREPARAÇÃO DOS DADOS (CRUCIAL: Igual ao Treino)
        self.test_data = self.raw_df.select_dtypes(include=[np.number])
        # Normalização Z-Score
        self.test_data = (self.test_data - self.test_data.mean()) / self.test_data.std()
        self.test_data = self.test_data.fillna(0).clip(-5, 5)
        
        self.results = {}
    
    def run_backtest(self, initial_balance=10000):
        print("🧪 INICIANDO BACKTEST DA IA GENESIS...")
        
        # Usa os últimos 20% para teste (Futuro)
        test_size = int(0.2 * len(self.test_data))
        test_df = self.test_data.tail(test_size).reset_index(drop=True)
        
        # Cria ambiente
        env = DummyVecEnv([lambda: FixedCryptoTradingEnv(test_df, initial_balance=initial_balance)])
        
        obs = env.reset()
        done = False
        
        equity_curve = [initial_balance]
        trades = []
        step = 0
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Captura info do ambiente
            net_worth = info[0]['net_worth']
            equity_curve.append(net_worth)
            
            # Registra Ação (Ignora Hold=0)
            act = action[0]
            if act != 0:
                trades.append({
                    'step': step,
                    'action': ["HOLD", "BUY", "SELL", "CLOSE"][act],
                    'reward': reward[0],
                    'equity': net_worth
                })
            step += 1
            
        self._analyze_results(equity_curve, trades, initial_balance)
        return self.results
    
    def _analyze_results(self, equity, trades, initial):
        total_ret = (equity[-1] - initial) / initial * 100
        wins = len([t for t in trades if t['reward'] > 0])
        total_ops = len(trades)
        win_rate = (wins / total_ops * 100) if total_ops > 0 else 0
        
        self.results = {
            'win_rate': win_rate,
            'total_return': total_ret,
            'total_trades': total_ops,
            'equity_curve': equity
        }

    def generate_report(self):
        print("\n📊 RELATÓRIO DE PERFORMANCE")
        print("="*30)
        print(f"📈 Retorno Total: {self.results['total_return']:.2f}%")
        print(f"🎯 Win Rate: {self.results['win_rate']:.1f}% ({self.results['total_trades']} trades)")
        
        try:
            plt.figure(figsize=(10,6))
            plt.plot(self.results['equity_curve'])
            plt.title("Curva de Patrimônio (Simulada)")
            plt.savefig("Genesis_AI/performance_chart.png")
            print("📉 Gráfico salvo em: Genesis_AI/performance_chart.png")
        except:
            print("⚠️ Matplotlib não instalado ou erro ao plotar.")