# Genesis_AI/train_genesis_v14.py (A VERDADE NUA E CRUA)
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import os

# CONFIGURAÇÃO RÍGIDA
DADOS_PATH = "../Binance/dataset_wld_clean.csv"
MODELO_PATH = "cerebros/genesis_wld_clean"
LOG_DIR = "logs_v14"
WINDOW_SIZE = 50 

def main():
    print("🧬 INICIANDO TREINO V14 (CONTROLO TOTAL)...")
    
    if not os.path.exists(DADOS_PATH):
        print("❌ Erro: Dataset não encontrado.")
        return

    # 1. Carrega Dados
    df_bruto = pd.read_csv(DADOS_PATH)
    
    # 2. Separa Preço Real
    df_price_real = df_bruto[['close']].copy()

    # 3. Prepara Features
    # Seleciona numéricos
    df_features = df_bruto.select_dtypes(include=[np.number]).copy()
    
    # REMOVE EXPLÍCITAMENTE
    cols_to_drop = ['target', 'timestamp', 'close']
    # Garante que só remove o que existe
    existing_drops = [c for c in cols_to_drop if c in df_features.columns]
    df_features = df_features.drop(columns=existing_drops)
    
    # IMPRIME AS FEATURES EXATAS
    print("\n🔧 FEATURES USADAS NO TREINO:")
    print(f"   Quantidade: {len(df_features.columns)}")
    print(f"   Lista: {list(df_features.columns)}")
    
    # Normaliza
    df_norm = (df_features - df_features.mean()) / df_features.std()
    df_norm = df_norm.fillna(0).clip(-5, 5)
    
    # CALCULA O SHAPE ESPERADO
    expected_shape = WINDOW_SIZE * len(df_features.columns)
    print(f"\n📏 SHAPE ESPERADO DA IA: {expected_shape} ({WINDOW_SIZE} x {len(df_features.columns)})")
    print("   (Se o erro no Live Trader pedir outro número, sabemos que o arquivo estava errado)\n")
    
    # 4. Cria Ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(df_norm, df_price_real, lookback_window=WINDOW_SIZE)])
    
    # 5. Treina
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log=LOG_DIR
    )
    
    print("🎯 Treinando (50k steps rápidos para teste)...")
    model.learn(total_timesteps=50000)
    
    model.save(MODELO_PATH)
    print(f"✅ MODELO V14 SALVO: {MODELO_PATH}")

if __name__ == "__main__":
    main()