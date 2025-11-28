# Genesis_AI/test_genesis_performance.py (CORRIGIDO E IMPORTANDO AMBIENTE CERTO)
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os

# --- IMPORTA O AMBIENTE ROBUSTO (QUE JÁ TEM PREÇO REAL + PROTEÇÃO) ---
# Certifique-se de que o arquivo fixed_trading_env.py existe na pasta Genesis_AI
try:
    from fixed_trading_env import RealisticTradingEnv
except ImportError:
    # Fallback se rodar de fora da pasta
    import sys
    sys.path.append(os.path.dirname(__file__))
    from fixed_trading_env import RealisticTradingEnv

class PerformanceTester:
    def __init__(self, model_path, test_data_path):
        # 1. Carregar Modelo
        if not os.path.exists(model_path + ".zip"):
             if os.path.exists("cerebros/" + model_path + ".zip"):
                 model_path = "cerebros/" + model_path
        
        print(f"🧠 Carregando modelo: {model_path}")
        self.model = PPO.load(model_path)
        
        # 2. Carregar Dados Brutos
        print(f"📚 Carregando dados: {test_data_path}")
        self.raw_df = pd.read_csv(test_data_path)
        
        # 3. PREPARAR PREÇO REAL (Atributo que faltava!)
        if 'close' in self.raw_df.columns:
            self.price_data_real = self.raw_df['close']
        else:
            self.price_data_real = self.raw_df.iloc[:, 0]

        # 4. PREPARAR DADOS NORMALIZADOS (Z-Score)
        df_num = self.raw_df.select_dtypes(include=[np.number])
        cols_drop = ['target', 'timestamp']
        if 'close' in df_num.columns: cols_drop.append('close')
            
        df_norm = df_num.drop(columns=[c for c in cols_drop if c in df_num.columns])
        
        self.test_data_norm = (df_norm - df_norm.mean()) / df_norm.std()
        self.test_data_norm = self.test_data_norm.fillna(0).clip(-5, 5)
        
        self.results = {}

    def run_backtest_with_env(self, env):
        """Roda o backtest injetando o ambiente configurado"""
        print("🧪 Executando Simulação de Mercado...")
        
        obs = env.reset()
        done = False
        
        try: initial_balance = env.envs[0].initial_balance
        except: initial_balance = 10000
            
        equity_curve = [initial_balance]
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            net_worth = info[0]['net_worth']
            equity_curve.append(net_worth)
            
        self._analyze_results(equity_curve, initial_balance)
        return self.results

    def run_backtest(self, initial_balance=10000):
        """Método legado (cria ambiente internamente se chamado direto)"""
        test_size = int(0.2 * len(self.test_data_norm))
        test_df_norm = self.test_data_norm.tail(test_size).reset_index(drop=True)
        test_price = self.price_data_real.tail(test_size).reset_index(drop=True)
        
        env = DummyVecEnv([lambda: RealisticTradingEnv(test_df_norm, test_price, initial_balance=initial_balance)])
        return self.run_backtest_with_env(env)

    def _analyze_results(self, equity, initial):
        total_ret = (equity[-1] - initial) / initial * 100
        peak = initial
        max_drawdown = 0
        for value in equity:
            if value > peak: peak = value
            if peak > 0:
                dd = (peak - value) / peak * 100
                if dd > max_drawdown: max_drawdown = dd
            
        self.results = {
            'total_return': total_ret,
            'max_drawdown': max_drawdown,
            'equity_curve': equity,
            'final_balance': equity[-1]
        }

    def generate_report(self):
        print("\n📊 RELATÓRIO DE PERFORMANCE V12")
        print("="*30)
        print(f"💰 Saldo Inicial: $10,000.00")
        print(f"💰 Saldo Final:   ${self.results['final_balance']:,.2f}")
        print(f"📈 Retorno Total: {self.results['total_return']:.2f}%")
        print(f"📉 Drawdown Máx:  {self.results['max_drawdown']:.2f}%")
        
        try:
            plt.figure(figsize=(10,6))
            plt.plot(self.results['equity_curve'])
            plt.title("Curva de Patrimônio (Teste Futuro)")
            plt.xlabel("Candles")
            plt.ylabel("Capital ($)")
            plt.grid(True, alpha=0.3)
            plt.savefig("Genesis_AI/performance_chart.png")
            print("📉 Gráfico salvo em: Genesis_AI/performance_chart.png")
        except: pass