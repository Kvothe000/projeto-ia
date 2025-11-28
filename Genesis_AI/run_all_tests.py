# Genesis_AI/run_all_tests.py (ALINHADO V12 - WINDOW 50)
import time
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from test_genesis_performance import PerformanceTester
from fixed_trading_env import RealisticTradingEnv

def run_comprehensive_test_suite():
    print("🧪 INICIANDO SUITE DE TESTES GENESIS")
    print("="*60)
    
    # --- ALINHAMENTO: DEVE SER IGUAL AO TREINO ---
    WINDOW_SIZE = 50 
    # ---------------------------------------------
    
    model_path = "cerebros/genesis_v12_final"
    data_path = "../Binance/dataset_v11_fusion.csv"
    
    try:
        # 1. Carrega Dados
        raw_df = pd.read_csv(data_path)
        
        # 2. Separa Preço Real
        price_data_real = raw_df[['close']].copy() # Mantém como DataFrame para compatibilidade
        
        # 3. Prepara Features (IGUAL AO TREINO)
        df_features = raw_df.select_dtypes(include=[np.number]).copy()
        cols_to_drop = ['target', 'timestamp', 'close']
        test_df_norm = df_features.drop(columns=[c for c in cols_to_drop if c in df_features.columns])
        
        # Normaliza
        test_df_norm = (test_df_norm - test_df_norm.mean()) / test_df_norm.std()
        test_df_norm = test_df_norm.fillna(0).clip(-5, 5)
        
        # 4. Recorte de Teste (Futuro)
        test_size = int(0.2 * len(test_df_norm))
        test_df_norm = test_df_norm.tail(test_size).reset_index(drop=True)
        price_data_real = price_data_real.tail(test_size).reset_index(drop=True)
        
        # 5. Ambiente de Teste
        env = DummyVecEnv([lambda: RealisticTradingEnv(test_df_norm, price_data_real, lookback_window=WINDOW_SIZE)])
        
        # 6. Execução
        print("\n1️⃣  EXECUTANDO BACKTEST DE PERFORMANCE...")
        # Usamos o PerformanceTester apenas para rodar o loop, injetando o ambiente configurado manualmente aqui
        tester = PerformanceTester(model_path, data_path)
        perf_results = tester.run_backtest_with_env(env)
        tester.generate_report()
        
        # Veredito
        print("\n🎯 VEREDITO FINAL:")
        ret = perf_results['total_return']
        if ret > 5.0: print("🌟 APROVADO: IA Lucrativa.")
        elif ret > 0: print("✅ REGULAR: Lucro Marginal.")
        else: print("❌ REPROVADO: Prejuízo.")
            
    except Exception as e:
        print(f"\n❌ Erro Crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_test_suite()