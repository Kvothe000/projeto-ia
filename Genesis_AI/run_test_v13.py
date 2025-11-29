# Genesis_AI/run_test_v13.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import matplotlib.pyplot as plt
import os
import numpy as np

def run_test():
    print("🧪 TESTANDO MODELO V13...")
    
    model_path = "cerebros/genesis_v13_corrected"
    data_path = "../Binance/dataset_v11_fusion.csv"
    
    # Carrega Dados
    raw = pd.read_csv(data_path)
    price = raw[['close']].copy()
    
    feat = raw.select_dtypes(include=[np.number])
    cols_drop = ['target', 'timestamp', 'close']
    feat = feat.drop(columns=[c for c in cols_drop if c in feat.columns])
    
    norm = (feat - feat.mean()) / feat.std()
    norm = norm.fillna(0).clip(-5, 5)
    
    # Teste nos últimos 20%
    test_size = int(0.2 * len(norm))
    test_norm = norm.tail(test_size).reset_index(drop=True)
    test_price = price.tail(test_size).reset_index(drop=True)
    
    # Ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(test_norm, test_price, lookback_window=30)])
    model = PPO.load(model_path)
    
    obs = env.reset()
    done = False
    equity = [10000]
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        equity.append(info[0]['net_worth'])
        
    print(f"💰 Saldo Final: ${equity[-1]:.2f}")
    
    plt.plot(equity)
    plt.title("Performance V13")
    plt.savefig("Genesis_AI/chart_v13.png")
    print("📉 Gráfico salvo.")

if __name__ == "__main__":
    run_test()