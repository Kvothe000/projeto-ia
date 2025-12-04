# Genesis_AI/test_veteran_on_2025.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import matplotlib.pyplot as plt
import os
import sys

# Adiciona path para importar classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_genesis_performance import PerformanceTester

# CONFIG
MODELO_PATH = "cerebros/genesis_wld_veteran" # O General
DADOS_PATH = "../Binance/dataset_2025.csv"   # O Campo de Batalha Novo
WINDOW_SIZE = 30 # O Veterano usa 30

def run_challenge():
    print("⚔️ DESAFIO: VETERANO vs. ANO 2025...")
    
    if not os.path.exists(DADOS_PATH):
        # Tenta caminhos alternativos
        if os.path.exists("dataset_2025.csv"): DADOS_PATH = "dataset_2025.csv"
        elif os.path.exists("Binance/dataset_2025.csv"): DADOS_PATH = "Binance/dataset_2025.csv"
        else: print("❌ Dataset 2025 não encontrado."); return

    # 1. Carrega Dados de 2025
    tester = PerformanceTester(MODELO_PATH, DADOS_PATH)
    
    # Usa o ano INTEIRO para teste (já que o Veterano treinou em dados antigos, 2025 é tudo novidade)
    df_norm = tester.test_data_norm
    price_real = tester.price_data_real
    
    print(f"📉 Simulando 1 Ano Completo ({len(df_norm)} candles)...")

    # 2. Ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(
        df_norm, 
        price_real, 
        initial_balance=10000, 
        lookback_window=WINDOW_SIZE
    )])
    
    # 3. Executa
    perf_results = tester.run_backtest_with_env(env)
    
    # 4. Relatório
    saldo_final = perf_results['final_balance']
    lucro_pct = perf_results['total_return']
    dd = perf_results['max_drawdown']
    
    print("="*40)
    print(f"📅 RESULTADO VETERANO EM 2025")
    print("="*40)
    print(f"💰 Saldo Final:   ${saldo_final:,.2f}")
    print(f"📈 Lucro Total:   {lucro_pct:.2f}%")
    print(f"📉 Drawdown Máx:  {dd:.2f}%")
    
    if lucro_pct > 0:
        print("🌟 SUCESSO: O Veterano lucrou em dados nunca vistos!")
    else:
        print("❌ FALHA: O mercado mudou demais, o Veterano não se adaptou.")

    # Gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(perf_results['equity_curve'], label="Patrimônio Veterano", color='blue')
    plt.axhline(y=10000, color='r', linestyle='--', label="Inicial")
    plt.title(f"Veterano vs 2025 (Lucro: {lucro_pct:.2f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("veteran_vs_2025.png")
    print("📉 Gráfico salvo: veteran_vs_2025.png")

if __name__ == "__main__":
    run_challenge()