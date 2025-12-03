# Genesis_AI/stress_test_genesis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import sys

# Importa o ambiente correto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fixed_trading_env import RealisticTradingEnv

class StressTester:
    def __init__(self, model_path, data_path="../Binance/dataset_wld_clean.csv"):
        self.model_path = model_path if model_path.endswith(".zip") else model_path + ".zip"
        self.data_path = data_path
        
    def _carregar_dados(self):
        if not os.path.exists(self.data_path):
            print("❌ Dataset não encontrado.")
            return None, None
            
        df = pd.read_csv(self.data_path)
        price_data = df['close'].values
        
        # Remove colunas não-features
        cols_drop = ['timestamp', 'close', 'target']
        df_obs = df.drop(columns=[c for c in cols_drop if c in df.columns])
        
        # Normalização (Z-Score Global)
        df_norm = (df_obs - df_obs.mean()) / df_obs.std()
        df_norm = df_norm.fillna(0).clip(-5, 5)
        
        return df_norm, price_data

    def run_market_crash_test(self):
        """Simula o comportamento em todo o histórico disponível (incluindo quedas)"""
        print("\n📉 EXECUTANDO TESTE DE CRASH (HISTÓRICO COMPLETO)...")
        
        df_norm, price_data = self._carregar_dados()
        if df_norm is None: return []

        # Usa JANELA 30 (Do modelo WLD) ou 50 (Do PEPE) - Ajuste conforme o modelo
        # Vamos tentar detectar ou usar padrão 30
        WINDOW = 30 
        
        env = DummyVecEnv([lambda: RealisticTradingEnv(df_norm, price_data, initial_balance=10000, lookback_window=WINDOW)])
        model = PPO.load(self.model_path)
        
        obs = env.reset()
        done = False
        trades_pct = []
        
        capital_atual = 10000
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Tenta extrair o resultado do trade se houve fechamento
            # Nota: O ambiente precisaria retornar o trade exato, mas podemos inferir pela variação do net_worth
            novo_capital = info[0]['net_worth']
            if novo_capital != capital_atual:
                delta_pct = (novo_capital - capital_atual) / capital_atual
                # Filtra variações pequenas (ruído de hold) vs Trades reais
                if abs(delta_pct) > 0.001: 
                    trades_pct.append(delta_pct)
                capital_atual = novo_capital
                
        return trades_pct

    def run_monte_carlo(self, trades_list, simulacoes=1000):
        """Embaralha os trades 1000 vezes para ver a probabilidade de falência"""
        print(f"\n🎲 RODANDO MONTE CARLO ({simulacoes} SIMULAÇÕES)...")
        
        if not trades_list:
            print("⚠️ Sem trades suficientes para Monte Carlo.")
            return

        capital_inicial = 10000
        resultados_finais = []
        falencias = 0
        
        plt.figure(figsize=(12, 6))
        
        for i in range(simulacoes):
            # Embaralha a ordem dos lucros/prejuízos
            np.random.shuffle(trades_list)
            
            curve = [capital_inicial]
            balance = capital_inicial
            quebrou = False
            
            for trade_pct in trades_list:
                balance *= (1 + trade_pct)
                curve.append(balance)
                if balance < capital_inicial * 0.5: # Considera falência se cair 50%
                    quebrou = True
            
            resultados_finais.append(balance)
            if quebrou: falencias += 1
            
            # Plota apenas as primeiras 50 para não poluir
            if i < 50:
                color = 'red' if curve[-1] < capital_inicial else 'green'
                plt.plot(curve, color=color, alpha=0.1)

        # Estatísticas
        prob_ruina = (falencias / simulacoes) * 100
        media_final = np.mean(resultados_finais)
        pior_cenario = np.min(resultados_finais)
        melhor_cenario = np.max(resultados_finais)
        
        print(f"🔥 Probabilidade de Ruína (Perder 50%): {prob_ruina:.2f}%")
        print(f"💰 Média Final Esperada: ${media_final:.2f}")
        print(f"💀 Pior Cenário (Azar Total): ${pior_cenario:.2f}")
        print(f"🚀 Melhor Cenário (Sorte Total): ${melhor_cenario:.2f}")
        
        plt.title(f"Monte Carlo Simulation (Ruína: {prob_ruina:.1f}%)")
        plt.ylabel("Capital ($)")
        plt.xlabel("Trades")
        plt.axhline(y=capital_inicial, color='black', linestyle='--')
        plt.savefig("Genesis_AI/monte_carlo_report.png")
        print("📉 Gráfico salvo: monte_carlo_report.png")

def main():
    # Testa o modelo WLD
    tester = StressTester("Genesis_AI/cerebros/genesis_wld_v2")
    
    # 1. Extrai lista de % de lucros/perdas de todo o histórico
    trades = tester.run_market_crash_test()
    print(f"📊 Trades Extraídos: {len(trades)}")
    
    if len(trades) > 10:
        # 2. Roda Monte Carlo
        tester.run_monte_carlo(trades)
    else:
        print("⚠️ Poucos trades para análise estatística robusta.")

if __name__ == "__main__":
    main()