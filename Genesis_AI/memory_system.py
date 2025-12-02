# Genesis_AI/memory_system.py (PADRONIZADO PARA 'CLOSE')
import pandas as pd
import os
from datetime import datetime

class MemorySystem:
    def __init__(self, filename="genesis_memory.csv"):
        self.memory_file = os.path.join(os.path.dirname(__file__), filename)
        self.buffer = []

    def memorizar(self, df_features, action, reward, preco_real):
        """
        Grava a experiência. Padronizamos 'preco_real' para 'close'
        para bater com o dataset de treino original.
        """
        try:
            # Converte features para dicionário
            registro = df_features.iloc[0].to_dict()
            
            # Metadados
            registro['action'] = action
            registro['reward'] = reward
            # --- CORREÇÃO: Nome da coluna padronizado ---
            registro['close'] = preco_real 
            # --------------------------------------------
            registro['timestamp'] = datetime.now().timestamp()
            
            self.buffer.append(registro)
            
            # Salva em blocos de 10
            if len(self.buffer) >= 10:
                self.consolidar_memoria()
                
        except Exception as e:
            print(f"⚠️ Erro ao memorizar: {e}")

    def consolidar_memoria(self):
        if not self.buffer: return
        
        df_new = pd.DataFrame(self.buffer)
        header = not os.path.exists(self.memory_file)
        
        # Modo append (adicionar ao final)
        df_new.to_csv(self.memory_file, mode='a', header=header, index=False)
        print(f"🧠 Memória consolidada: {len(self.buffer)} registros.")
        self.buffer = []

    def carregar_memoria_recente(self):
        if os.path.exists(self.memory_file):
            try:
                return pd.read_csv(self.memory_file)
            except: return None
        return None