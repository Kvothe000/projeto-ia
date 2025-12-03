# Genesis_AI/run_test_pepe.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os
import sys

# Garante que encontra o ambiente
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fixed_trading_env import RealisticTradingEnv

# CONFIGURAÇÃO
MODELO_PATH = "Genesis_AI/cerebros/genesis_pepe_v1"
DADOS_PATH = "../Binance/dataset_pepe_clean.csv"
WINDOW_SIZE = 50 # O treino da PEPE usou 50

def run():
    print("🐸 TESTANDO GÊNESIS PEPE (Janela 50)...")
    
    if not os.path.exists(DADOS_PATH):
        print(f"❌ Erro: Dataset {DADOS_PATH} não encontrado.")
        return

    if not os.path.exists(MODELO_PATH + ".zip"):
        print(f"❌ Erro: Modelo {MODELO_PATH} não encontrado.")
        return

    # 1. Carrega e Prepara
    model = PPO.load(MODELO_PATH)
    df = pd.read_csv(DADOS_PATH)
    
    # Separa Teste (Últimos 20%)
    split = int(len(df) * 0.8)
    df_test = df.iloc[split:].reset_index(drop=True)
    
    price_data = df_test['close'].values
    
    # Features
    cols_drop = ['timestamp', 'close', 'target']
    df_obs = df_test.drop(columns=[c for c in cols_drop if c in df_test.columns])
    
    # Normalização (Z-Score)
    df_norm = (df_obs - df_obs.mean()) / df_obs.std()
    df_norm = df_norm.fillna(0).clip(-5, 5)
    
    # 2. Ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(
        df_norm, 
        price_data, 
        initial_balance=10000,
        lookback_window=WINDOW_SIZE
    )])
    
    obs = env.reset()
    done = False
    equity = [10000]
    
    print("📉 Executando simulação de alta volatilidade...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        equity.append(info[0]['net_worth'])
        
    saldo_final = equity[-1]
    lucro_pct = (saldo_final - 10000) / 10000 * 100
    
    print("-" * 30)
    print(f"💰 Saldo Final: ${saldo_final:.2f}")
    print(f"📈 Lucro Total: {lucro_pct:.2f}%")
    print("-" * 30)
    
    # Gráfico
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(equity, label="Patrimônio PEPE", color='green')
        plt.axhline(y=10000, color='r', linestyle='--', label="Inicial")
        plt.title(f"Performance PEPE Specialist (Lucro: {lucro_pct:.2f}%)")
        plt.xlabel("Trades/Candles")
        plt.ylabel("Capital ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("pepe_result.png")
        print("📉 Gráfico salvo: pepe_result.png")
    except:
        print("⚠️ Gráfico não gerado (Matplotlib erro).")

if __name__ == "__main__":
    run()