# Genesis_AI/test_genesis_performance.py (LIMPO E VINCULADO)
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os
import sys

# Adiciona o diretório atual ao path para garantir a importação
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- IMPORTA O AMBIENTE ROBUSTO (A FONTE DA VERDADE) ---
try:
    from fixed_trading_env import RealisticTradingEnv
except ImportError:
    print("❌ ERRO CRÍTICO: Não encontrei 'fixed_trading_env.py'.")
    print("   Certifique-se de que este arquivo está na pasta Genesis_AI.")
    exit()

class PerformanceTester:
    def __init__(self, model_path, test_data_path):
        # 1. Carrega Modelo
        if not os.path.exists(model_path + ".zip"):
             if os.path.exists("cerebros/" + model_path + ".zip"):
                 model_path = "cerebros/" + model_path
        
        print(f"🧠 Carregando modelo: {model_path}")
        self.model = PPO.load(model_path)
        
        # 2. Carrega e Prepara Dados
        print(f"📚 Carregando dados: {test_data_path}")
        self.raw_df = pd.read_csv(test_data_path)
        
        # Separa Preço Real (Vital para cálculo de lucro)
        # Tenta 'close', se não tiver usa a primeira coluna
        if 'close' in self.raw_df.columns:
            self.price_data_real = self.raw_df['close']
        else:
            self.price_data_real = self.raw_df.iloc[:, 0] # Fallback perigoso, mas evita crash

        # Prepara Dados Normalizados (Visão da IA)
        df_num = self.raw_df.select_dtypes(include=[np.number])
        cols_drop = ['target', 'timestamp', 'close']
        # Remove apenas o que existe
        cols_present = [c for c in cols_drop if c in df_num.columns]
        df_norm = df_num.drop(columns=cols_present)
        
        # Normalização Z-Score
        self.test_data_norm = (df_norm - df_norm.mean()) / df_norm.std()
        self.test_data_norm = self.test_data_norm.fillna(0).clip(-5, 5)
        
        self.results = {}

    def run_backtest_with_env(self, env):
        """Roda a simulação usando o ambiente configurado externamente"""
        print("🧪 Executando Simulação de Mercado...")
        
        obs = env.reset()
        done = False
        
        # Tenta pegar saldo inicial do ambiente
        try: initial_balance = env.envs[0].initial_balance
        except: initial_balance = 10000
            
        equity_curve = [initial_balance]
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Captura patrimônio
            net_worth = info[0]['net_worth']
            equity_curve.append(net_worth)
            
        self._analyze_results(equity_curve, initial_balance)
        return self.results

    def _analyze_results(self, equity, initial):
        final_val = equity[-1]
        total_ret = (final_val - initial) / initial * 100
        
        # Drawdown
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
            'final_balance': final_val
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