# Genesis_AI/train_genesis.py (VERSÃO CORRIGIDA PARA WINDOWS)
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv # <--- ESSENCIAL
from sb3_contrib import RecurrentPPO
from crypto_env_advanced import AdvancedCryptoTradingEnv
import os

# Config
DADOS_PATH = "../Binance/dataset_v11_fusion.csv"
MODELO_PATH = "cerebros/genesis_lstm_v1"
LOG_DIR = "logs"

def treinar_genesis():
    print("🧬 INICIANDO GÊNESIS (MODO SINGLE PROCESS)...")
    
    if not os.path.exists(DADOS_PATH):
        print("❌ Erro: Dataset não encontrado. Rode o gerar_dataset_v11_fusion.py primeiro!")
        return
        
    # Carrega e Limpa
    df = pd.read_csv(DADOS_PATH)
    df = df.select_dtypes(include=['float64', 'int64'])
    if 'target' in df.columns: df = df.drop(columns=['target'])
    
    print(f"📚 Memória Carregada: {len(df)} momentos de mercado.")

    # CRIA AMBIENTE (DummyVecEnv para evitar erro no Windows)
    env = DummyVecEnv([lambda: AdvancedCryptoTradingEnv(df)])

    print("🧠 Instanciando Rede Neural Recorrente (LSTM)...")
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        gamma=0.995,
        tensorboard_log=LOG_DIR
    )

    print("🎓 Iniciando treinamento intensivo...")
    try:
        model.learn(total_timesteps=1_000_000, progress_bar=True)
        model.save(MODELO_PATH)
        print(f"✅ TREINO CONCLUÍDO! Cérebro salvo em {MODELO_PATH}")
    except Exception as e:
        print(f"❌ Erro durante o treino: {e}")

if __name__ == "__main__":
    treinar_genesis()