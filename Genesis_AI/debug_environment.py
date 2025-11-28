# Genesis_AI/debug_environment.py
import pandas as pd
import numpy as np
from fixed_trading_env import RealisticTradingEnv

def debug_environment():
    print("🔧 DEBUG DO AMBIENTE DE TRADING")
    print("="*50)
    
    # Carrega dados pequenos para teste
    df = pd.read_csv("../Binance/dataset_v11_fusion.csv").head(1000)
    price_data = df['close'].values
    
    # Features normalizadas
    df_features = df.select_dtypes(include=[np.number]).copy()
    cols_to_drop = ['target', 'timestamp', 'close']
    df_features = df_features.drop(columns=[c for c in cols_to_drop if c in df_features.columns])
    df_norm = (df_features - df_features.mean()) / df_features.std()
    df_norm = df_norm.fillna(0)
    
    # Cria ambiente
    env = RealisticTradingEnv(df_norm, price_data, lookback_window=50)
    
    print("🧪 TESTANDO AÇÕES MANUAIS:")
    obs = env.reset()
    
    for i in range(10):
        # Testa ações diferentes
        for action in [0, 1, 2]:  # HOLD, BUY, SELL
            next_obs, reward, done, info = env.step(action)
            print(f"Step {i}, Action {action}:")
            print(f"  Reward: {reward}")
            print(f"  Portfolio: {env.portfolio_value:.2f}")
            print(f"  Balance: {env.balance:.2f}")
            print(f"  Position: {env.position}")
            print(f"  Done: {done}")
            print("-" * 30)
            
            if done:
                env.reset()
    
    print("🎯 DIAGNÓSTICO FINAL:")
    print(f"Portfolio Final: {env.portfolio_value:.2f}")
    print(f"Retorno: {(env.portfolio_value - 10000) / 10000 * 100:.2f}%")

if __name__ == "__main__":
    debug_environment()