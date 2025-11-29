# Genesis_AI/run_test_wld.py (CORRIGIDO: WINDOW 50)
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import matplotlib.pyplot as plt
import os

def run():
    print("🧪 TESTANDO GÊNESIS WLD (Janela 50)...")
    
    model_path = "Genesis_AI/cerebros/genesis_wld_clean"
    data_path = "../Binance/dataset_wld_clean.csv"

    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Erro: Modelo {model_path} não encontrado.")
        return

    model = PPO.load(model_path)
    df = pd.read_csv(data_path)
    
    # Separa os últimos 20% para teste
    split = int(len(df) * 0.8)
    df_test = df.iloc[split:].reset_index(drop=True)
    
    price_data = df_test['close'].values
    
    # Prepara Features (Remove colunas não usadas)
    cols_drop = ['timestamp', 'close', 'target']
    df_obs = df_test.drop(columns=[c for c in cols_drop if c in df_test.columns])
    
    # Normaliza
    df_norm = (df_obs - df_obs.mean()) / df_obs.std()
    df_norm = df_norm.fillna(0).clip(-5, 5)
    
    # --- CORREÇÃO AQUI: Janela de 50 para bater com o treino ---
    env = DummyVecEnv([lambda: RealisticTradingEnv(
        df_norm, 
        price_data, 
        lookback_window=50
    )])
    # -----------------------------------------------------------
    
    obs = env.reset()
    done = False
    equity = [10000]
    
    print("📉 Executando simulação...")
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
    plt.figure(figsize=(10, 6))
    plt.plot(equity, label="Patrimônio")
    plt.axhline(y=10000, color='r', linestyle='--', label="Inicial")
    plt.title(f"Performance WLD Specialist (Lucro: {lucro_pct:.2f}%)")
    plt.xlabel("Trades/Candles")
    plt.ylabel("Capital ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    try:
        plt.savefig("wld_result.png")
        print("📉 Gráfico salvo: wld_result.png")
    except:
        print("⚠️ Não foi possível salvar o gráfico.")

if __name__ == "__main__":
    run()