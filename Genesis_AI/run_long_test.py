# Genesis_AI/run_long_test.py (COM RELATÓRIO DE ERROS)
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os
import sys

# Importa ambiente
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fixed_trading_env import RealisticTradingEnv

# CONFIG
MODELO_PATH = "Genesis_AI/cerebros/genesis_wld_veteran"
DADOS_PATH = "../Binance/dataset_wld_1ano.csv"
WINDOW_SIZE = 30

def run_long_test():
    print("⏳ INICIANDO TESTE DE STRESS (DIAGNÓSTICO COMPLETO)...")
    
    if not os.path.exists(DADOS_PATH):
        print(f"❌ Erro: Dataset {DADOS_PATH} não encontrado.")
        return

    model = PPO.load(MODELO_PATH)
    df = pd.read_csv(DADOS_PATH)
    
    # Separa Preço Real e Features
    price_data = df['close'].values
    cols_drop = ['timestamp', 'close', 'target']
    df_obs = df.drop(columns=[c for c in cols_drop if c in df.columns])
    
    # Normaliza (Global)
    df_norm = (df_obs - df_obs.mean()) / df_obs.std()
    df_norm = df_norm.fillna(0).clip(-5, 5)
    
    # Ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(
        df_norm, 
        price_data, 
        initial_balance=10000,
        lookback_window=WINDOW_SIZE
    )])
    
    obs = env.reset()
    done = False
    equity = [10000]
    trades_log = [] # Lista para gravar os erros
    
    capital_anterior = 10000
    passo = 0
    
    print("📉 Simulando 1 ano candle a candle...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        
        net_worth = info[0]['net_worth']
        equity.append(net_worth)
        
        # Detecta Trade (Variação brusca de saldo ou ação de fechar)
        # Simplificação: Se a ação for BUY/SELL/CLOSE, registramos
        act = action.item()
        
        # Registra o estado a cada hora (4 candles) ou se houver trade
        if act != 0 or passo % 4 == 0:
            trades_log.append({
                "step": passo,
                "preco": price_data[passo] if passo < len(price_data) else 0,
                "acao": ["HOLD", "BUY", "SELL", "CLOSE"][act],
                "saldo": net_worth,
                "lucro_acumulado": (net_worth - 10000)
            })
        
        passo += 1
        
    # Relatório Final
    saldo_final = equity[-1]
    lucro_pct = (saldo_final - 10000) / 10000 * 100
    
    print("="*40)
    print(f"💰 Saldo Final: ${saldo_final:,.2f}")
    print(f"📉 Lucro Total: {lucro_pct:.2f}%")
    print("="*40)

    # Salva o Log de Erros
    pd.DataFrame(trades_log).to_csv("Genesis_AI/debug_long_run.csv", index=False)
    print("📄 Relatório de trades salvo em: Genesis_AI/debug_long_run.csv")

    # Gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label="Patrimônio", linewidth=1, color='red')
    plt.axhline(y=10000, color='blue', linestyle='--', label="Inicial")
    plt.title(f"Stress Test 1 Ano (Drawdown: {lucro_pct:.2f}%)")
    plt.savefig("long_run_result.png")
    print("📉 Gráfico salvo: long_run_result.png")

if __name__ == "__main__":
    run_long_test()