# Genesis_AI/train_genesis_corrected.py
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from fixed_trading_env import RealisticTradingEnv
import os

class TrainingValidator(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingValidator, self).__init__(verbose)
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if self.n_calls % 5000 == 0:
            # Salva checkpoint periódico
            self.model.save(f"cerebros/genesis_checkpoint_{self.n_calls}")
            print(f"💾 Checkpoint {self.n_calls} salvo.")
        return True

def main():
    print("🧠 INICIANDO TREINAMENTO CORRIGIDO DO GENESIS...")
    
    try:
        df = pd.read_csv('../Binance/dataset_v11_fusion.csv')
        # Remove colunas não numéricas
        cols_to_drop = ['target', 'timestamp'] 
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        df = df.select_dtypes(include=[np.number])
        
        print(f"📊 Dados carregados: {df.shape}")
        
    except Exception as e:
        print(f"❌ Erro dados: {e}")
        return
    
    # Cria ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(df)])
    
    # Configuração HIPER-CONSERVADORA (Para não explodir)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,  # Muito lento (Segurança)
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        clip_range=0.1,      # Conservador
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/genesis_corrected"
    )
    
    print("🎯 Iniciando Treino Seguro (50k steps)...")
    validator = TrainingValidator()
    
    model.learn(total_timesteps=50000, callback=validator)
    
    model.save("cerebros/genesis_corrected")
    print("✅ Modelo Salvo: cerebros/genesis_corrected")

if __name__ == "__main__":
    main()