# Genesis_AI/train_genesis_wld_v2.py (CORRIGIDO)
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import os

# CONFIGURAÇÃO
MODELO = "Genesis_AI/cerebros/genesis_wld_v2"
LOG_DIR = "logs_wld_v2"
WINDOW = 30 

def main():
    print("🧬 TREINANDO GÊNESIS WLD V2 (META 1% + CLOSE ACTION)...")
    
    # --- CORREÇÃO DE CAMINHO ROBUSTA ---
    # Procura o arquivo em vários lugares possíveis
    base_dir = os.path.dirname(os.path.abspath(__file__))
    caminhos_possiveis = [
        os.path.join(base_dir, "..", "dataset_wld_clean.csv"), # Padrão
        os.path.join(base_dir, "dataset_wld_clean.csv"),                 # Local
        "dataset_wld_clean.csv",                                          # Raiz
        "Binance/dataset_wld_clean.csv"                                   # Relativo
    ]
    
    dados_path = None
    for p in caminhos_possiveis:
        if os.path.exists(p):
            dados_path = p
            break
            
    if not dados_path:
        print("❌ Erro: Dataset 'dataset_wld_clean.csv' não encontrado.")
        print("   Certifique-se de rodar 'python Binance/gerar_dataset_wld.py' primeiro.")
        return
        
    print(f"📚 Carregando dados de: {dados_path}")
    df = pd.read_csv(dados_path)
    print(f"📊 Registros carregados: {len(df)}")
    
    # Prepara Dados
    # O ambiente precisa do preço real separado
    price_data = df['close'].values
    
    # Features para a IA (Remove colunas que não são input)
    cols_drop = ['timestamp', 'close', 'target'] 
    # Garante que só dropa o que existe
    cols_to_drop = [c for c in cols_drop if c in df.columns]
    df_obs = df.drop(columns=cols_to_drop)
    
    # Normalização Z-Score
    df_norm = (df_obs - df_obs.mean()) / df_obs.std()
    df_norm = df_norm.fillna(0).clip(-5, 5)
    
    # Ambiente com Bônus de 1% (V2)
    # Certifique-se de que o fixed_trading_env.py já foi atualizado para V2!
    env = DummyVecEnv([lambda: RealisticTradingEnv(
        df_norm, 
        price_data, 
        initial_balance=10000,
        lookback_window=WINDOW
    )])
    
    # Modelo PPO
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
    
    print("🏋️ Iniciando Treino de Precisão...")
    # Treinamos por 150k passos para ela ter tempo de descobrir o bônus
    model.learn(total_timesteps=150000)
    
    model.save(MODELO)
    print(f"✅ CÉREBRO WLD V2 SALVO: {MODELO}")
    print("👉 Agora atualize o 'live_trader_wld.py' para usar o modelo 'genesis_wld_v2'.")

if __name__ == "__main__":
    main()