# Genesis_AI/train_genesis_2025.py (CORRIGIDO)
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import os

# CONFIG
NOME_ARQUIVO = "dataset_2025.csv"
MODELO_PATH = "Genesis_AI/cerebros/genesis_2025"
LOG_DIR = "logs_2025"
WINDOW_SIZE = 30

def main():
    print("🧬 INICIANDO TREINO GÊNESIS 2025 (75% TREINO / 25% TESTE)...")
    
    # --- BUSCA ROBUSTA DO ARQUIVO ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    caminhos_possiveis = [
        os.path.join(base_dir, "..", "Binance", NOME_ARQUIVO), # ../Binance/dataset...
        os.path.join(base_dir, NOME_ARQUIVO),                 # ./dataset...
        NOME_ARQUIVO,                                         # dataset...
        f"Binance/{NOME_ARQUIVO}"                             # Binance/dataset...
    ]
    
    dados_path = None
    for p in caminhos_possiveis:
        if os.path.exists(p):
            dados_path = p
            break
            
    if not dados_path:
        print(f"❌ Erro: Dataset '{NOME_ARQUIVO}' não encontrado.")
        print("   Certifique-se de rodar 'python Binance/gerar_dataset_2025.py' primeiro.")
        return

    print(f"📚 Carregando dados de: {dados_path}")
    df = pd.read_csv(dados_path)
    
    # --- O CORTE ESTRATÉGICO (SPLIT) ---
    corte = int(len(df) * 0.75)
    df_treino = df.iloc[:corte].reset_index(drop=True)
    # Os últimos 25% ficam guardados para o teste depois
    
    print(f"📊 Total Registros: {len(df)}")
    print(f"🏋️ Dados de Treino: {len(df_treino)} (Jan-Set)")
    
    # Prepara Dados de Treino
    # O ambiente precisa do preço real separado
    price_data = df_treino['close'].values
    
    cols_drop = ['timestamp', 'close', 'target'] 
    # Garante que remove apenas o que existe
    cols_existentes = [c for c in cols_drop if c in df_treino.columns]
    df_obs = df_treino.drop(columns=cols_existentes)
    
    # Normalização (Aprende a média APENAS no treino para não viciar)
    df_norm = (df_obs - df_obs.mean()) / df_obs.std()
    df_norm = df_norm.fillna(0).clip(-5, 5)
    
    # Ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(
        df_norm, 
        price_data, 
        initial_balance=10000, 
        lookback_window=WINDOW_SIZE
    )])
    
    # Modelo
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048, 
        batch_size=64, 
        gamma=0.99,
        tensorboard_log=LOG_DIR
    )
    
    print("🔥 Treinando...")
    model.learn(total_timesteps=200000)
    
    model.save(MODELO_PATH)
    print(f"✅ CÉREBRO 2025 SALVO: {MODELO_PATH}")

if __name__ == "__main__":
    main()