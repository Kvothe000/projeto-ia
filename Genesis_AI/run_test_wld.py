# Genesis_AI/run_test_wld.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import matplotlib.pyplot as plt
import os

def run():
    print("🧪 TESTANDO GÊNESIS WLD...")
    
    model = PPO.load("cerebros/genesis_wld_v1")
    df = pd.read_csv("../Binance/dataset_wld_clean.csv")
    
    # Separa os últimos 20% para teste
    split = int(len(df) * 0.8)
    df_test = df.iloc[split:].reset_index(drop=True)
    
    price_data = df_test['close'].values
    
    cols_drop = ['timestamp', 'close', 'target']
    df_obs = df_test.drop(columns=[c for c in cols_drop if c in df_test.columns])
    df_norm = (df_obs - df_obs.mean()) / df_obs.std()
    df_norm = df_norm.fillna(0).clip(-5, 5)
    
    env = DummyVecEnv([lambda: RealisticTradingEnv(df_norm, price_data, lookback_window=30)])
    
    obs = env.reset()
    done = False
    equity = [10000]
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        equity.append(info[0]['net_worth'])
        
    print(f"💰 Saldo Final: ${equity[-1]:.2f}")
    
    # Gráfico Rápido
    plt.figure()
    plt.plot(equity)
    plt.title("WLD Specialist Performance")
    plt.savefig("wld_result.png")
    print("📉 Gráfico salvo: wld_result.png")

if __name__ == "__main__":
    run()