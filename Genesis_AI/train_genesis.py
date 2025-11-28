# Genesis_AI/train_genesis.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO # <--- LSTM (Memória)
from crypto_env_advanced import AdvancedCryptoTradingEnv
import os

# Config
DADOS_PATH = "../Binance/dataset_v11_fusion.csv" # Usa nossos melhores dados
MODELO_PATH = "cerebros/genesis_lstm_v1"
LOG_DIR = "logs"

def treinar_genesis():
    print("🧬 INICIANDO PROJETO GÊNESIS (LSTM + PPO)...")
    
    # 1. Carrega Memória Histórica
    if not os.path.exists(DADOS_PATH):
        print("❌ Erro: Gere o dataset V11 primeiro na pasta Binance!")
        return
        
    df = pd.read_csv(DADOS_PATH)
    # Limpeza: A IA só come números
    df = df.select_dtypes(include=['float64', 'int64'])
    if 'target' in df.columns: df = df.drop(columns=['target'])
    
    print(f"📚 Memória Carregada: {len(df)} momentos de mercado.")

    # 2. Cria o Ambiente (Multi-Processado para velocidade)
    # Criamos 4 clones da IA para aprenderem em paralelo
    env = SubprocVecEnv([lambda: AdvancedCryptoTradingEnv(df) for _ in range(4)])

    # 3. O Cérebro (Recurrent PPO)
    # MlpLstmPolicy = Cérebro com Memória de Curto Prazo
    print("🧠 Instanciando Rede Neural Recorrente (LSTM)...")
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=128,
        gamma=0.995, # Visão de longo prazo
        tensorboard_log=LOG_DIR
    )

    # 4. Educação Intensiva
    print("🎓 Iniciando treinamento intensivo (1 Milhão de Steps)...")
    model.learn(total_timesteps=1_000_000)
    
    # 5. Salvar
    model.save(MODELO_PATH)
    print(f"💾 Gênesis V1 Salva! Cérebro guardado em {MODELO_PATH}")

if __name__ == "__main__":
    treinar_genesis()