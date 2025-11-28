# Genesis_AI/treinar_agente.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from crypto_env import CryptoTradingEnv
import os

# Carrega nossos dados ricos (V11 Fusion)
# Nota: A IA precisa de dados LIMPOS. O dataset_v11_fusion.csv é perfeito.
try:
    df = pd.read_csv('../Binance/dataset_v11_fusion.csv')
    # Remove colunas não numéricas se houver (timestamp string, etc)
    df = df.select_dtypes(include=['float64', 'int64'])
    # Remove o Target (A IA vai descobrir o target sozinha pela recompensa!)
    if 'target' in df.columns:
        df = df.drop(columns=['target'])
except:
    print("❌ Gere o dataset V11 primeiro!")
    exit()

print(f"🧠 Inicializando Projeto Gênesis com {len(df)} memórias de mercado...")

# Cria a Arena
env = DummyVecEnv([lambda: CryptoTradingEnv(df)])

# Cria o Agente (O Sócio)
# MlpPolicy = Rede Neural Padrão
# verbose=1 = Mostra o que está pensando
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=0.01)

print("🏋️ Começando o treino intensivo...")
print("   A IA vai simular milhares de trades e aprender com os erros.")

# Treina por 1 milhão de passos (simulados)
model.learn(total_timesteps=100000)

# Salva o Cérebro
os.makedirs("cerebros", exist_ok=True)
model.save("cerebros/genesis_v1")
print("✅ Cérebro Gênesis V1 salvo com sucesso!")