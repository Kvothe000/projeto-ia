# Genesis_AI/train_genesis.py (VERSÃO WIN-COMPATIBLE)
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv # <--- MUDANÇA IMPORTANTE
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
        print("❌ Erro: Dataset não encontrado.")
        return
        
    df = pd.read_csv(DADOS_PATH)
    
    # LIMPEZA CRÍTICA
    # Remove colunas de texto e garante que 'close' está lá
    df = df.select_dtypes(include=['float64', 'int64'])
    
    # Remove target antigo se existir (para não viciar a IA)
    if 'target' in df.columns: df = df.drop(columns=['target'])
    
    print(f"📚 Dados carregados: {len(df)} linhas. Colunas: {len(df.columns)}")

    # CRIA O AMBIENTE (Processo Único para evitar erro no Windows)
    # env = SubprocVecEnv(...) -> CAUSA ERRO NO WINDOWS
    env = DummyVecEnv([lambda: AdvancedCryptoTradingEnv(df)])

    print("🧠 Instanciando Rede Neural LSTM...")
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024, # Steps menores para feedback mais rápido
        batch_size=64,
        gamma=0.99,
        tensorboard_log=LOG_DIR
    )

    print("🎓 Iniciando treino (Isso vai demorar um pouco)...")
    try:
        model.learn(total_timesteps=1_000_000, progress_bar=True)
        model.save(MODELO_PATH)
        print(f"✅ TREINO CONCLUÍDO! Cérebro salvo em {MODELO_PATH}")
    except Exception as e:
        print(f"❌ Erro durante o treino: {e}")

if __name__ == "__main__":
    treinar_genesis()