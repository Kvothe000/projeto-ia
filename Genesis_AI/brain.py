# Genesis_AI/brain.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from market_env import CryptoGenesisEnv
import pandas as pd

class GenesisBrain:
    def __init__(self, dataset_path, model_path="cerebros/genesis_v1"):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None
        
        # Carrega dados (Memória Histórica)
        try:
            self.df = pd.read_csv(dataset_path)
            # Limpeza para garantir que a IA só vê números
            self.df = self.df.select_dtypes(include=['float64', 'int64'])
            if 'target' in self.df.columns:
                self.df = self.df.drop(columns=['target']) # A IA não precisa de gabarito, ela aprende vivendo
            if 'timestamp' in self.df.columns:
                self.df = self.df.drop(columns=['timestamp'])
            print(f"🧠 Memória Carregada: {len(self.df)} momentos de mercado.")
        except Exception as e:
            print(f"❌ Erro ao carregar memória: {e}")
            exit()

    def criar_ambiente(self):
        """Cria a Arena de Treino"""
        return DummyVecEnv([lambda: CryptoGenesisEnv(self.df)])

    def nascer(self):
        """Inicializa uma nova IA do zero"""
        print("🐣 Gênesis está nascendo...")
        env = self.criar_ambiente()
        
        # Configuração da Rede Neural (MlpPolicy)
        self.model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003, 
            n_steps=2048, 
            batch_size=64, 
            gamma=0.99, # Fator de desconto (visão de longo prazo)
            ent_coef=0.01 # Curiosidade (explorar novas estratégias)
        )
        print("✨ IA Inicializada e pronta para aprender.")

    def treinar(self, passos=100000):
        """Ciclo de Estudo Intensivo"""
        if self.model is None: self.nascer()
        
        print(f"📚 Iniciando sessão de estudo ({passos} passos)...")
        self.model.learn(total_timesteps=passos)
        self.salvar()

    def salvar(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"💾 Conhecimento salvo em: {self.model_path}")

    def carregar(self):
        if os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path)
            print("🧠 Conhecimento anterior restaurado.")
            return True
        return False