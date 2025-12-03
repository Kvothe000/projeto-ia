# Genesis_AI/run_all_tests.py (CONFIGURADO PARA WLD V2)
import time
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# Importa as classes necessárias
from test_genesis_performance import PerformanceTester
from fixed_trading_env import RealisticTradingEnv

def run_comprehensive_test_suite():
    print("🧪 INICIANDO SUITE DE TESTES GENESIS (WLD V2)")
    print("="*60)
    
    # --- CONFIGURAÇÃO CRÍTICA DO WLD V2 ---
    WINDOW_SIZE = 30   # O modelo WLD foi treinado com 30 candles
    
    # Caminhos atualizados para o novo cérebro e dados WLD
    model_path = "Genesis_AI/cerebros/genesis_wld_veteran"
    data_path = "../Binance/dataset_wld_clean.csv"
    
    # Verificação de segurança do caminho dos dados
    if not os.path.exists(data_path):
        if os.path.exists("dataset_wld_clean.csv"):
            data_path = "dataset_wld_clean.csv"
        elif os.path.exists("Binance/dataset_wld_clean.csv"):
            data_path = "Binance/dataset_wld_clean.csv"
    
    try:
        # 1. Prepara Dados para o Teste
        print(f"📚 Carregando dados de: {data_path}")
        
        # Usamos o Tester para carregar e normalizar os dados corretamente
        tester = PerformanceTester(model_path, data_path)
        
        # 2. Separa o Conjunto de Teste (Futuro)
        # Usa os últimos 20% dos dados que a IA nunca viu no treino
        test_size = int(0.2 * len(tester.test_data_norm))
        
        df_teste_norm = tester.test_data_norm.tail(test_size).reset_index(drop=True)
        price_teste_real = tester.price_data_real.tail(test_size).reset_index(drop=True)
        
        print(f"📉 Testando em {len(df_teste_norm)} candles desconhecidos (Futuro)...")

        # 3. Cria Ambiente REAL (Com a Janela 30 correta)
        # Importante: Passamos dados normalizados para a IA ver + Preço real para calcular lucro
        env = DummyVecEnv([lambda: RealisticTradingEnv(
            df_teste_norm, 
            price_teste_real, 
            initial_balance=10000, 
            lookback_window=WINDOW_SIZE
        )])
        
        # 4. Roda a Simulação
        print("\n1️⃣  EXECUTANDO BACKTEST DE PERFORMANCE...")
        perf_results = tester.run_backtest_with_env(env)
        tester.generate_report()
        
        # 5. Veredito Final
        saldo_final = perf_results['final_balance']
        lucro_pct = perf_results['total_return']
        
        print("\n🎯 VEREDITO FINAL:")
        if saldo_final > 10000:
            print(f"✅ APROVADO: A IA gerou lucro de {lucro_pct:.2f}%!")
            if lucro_pct > 10:
                print("🌟 EXCELENTE: Resultado muito acima da média.")
        elif saldo_final > 5000:
            print(f"⚠️ REGULAR: Prejuízo de {lucro_pct:.2f}%, mas o Stop Loss protegeu a banca.")
        else:
            print(f"❌ REPROVADO: Falência técnica ({lucro_pct:.2f}%).")

    except Exception as e:
        print(f"\n❌ Erro Crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_test_suite()