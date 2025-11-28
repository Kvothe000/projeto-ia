# Genesis_AI/quick_validation.py (SHAPE CORRIGIDO FINAL)
import numpy as np
from stable_baselines3 import PPO
import os
import pandas as pd

def quick_validation():
    print("🎯 VALIDAÇÃO RÁPIDA (SHAPE CORRIGIDO)...")
    
    model_path = "cerebros/genesis_v2_stable"
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Erro: Modelo não encontrado em {model_path}")
        return

    try:
        model = PPO.load(model_path)
        print(f"✅ Modelo carregado: {model_path}")
        
        # --- CORREÇÃO AQUI ---
        # A IA foi treinada para ver: (30 velas x 11 Features)
        N_FEATURES = 11 
        WINDOW_SIZE = 30 
        # ---------------------
        
        dummy_obs = np.random.normal(0, 1, (WINDOW_SIZE, N_FEATURES)).astype(np.float32)
        
        print(f"🤖 Testando com input shape: ({WINDOW_SIZE}, {N_FEATURES})...")
        actions = []
        
        for i in range(10):
            # Gera novo ruído com o shape correto
            obs_ruido = np.random.normal(0, 1, (WINDOW_SIZE, N_FEATURES)).astype(np.float32)
            action, _ = model.predict(obs_ruido)
            
            # Extrai o número inteiro
            actions.append(action.item())
            
        print(f"Ações tomadas: {actions}")
        
        unique_actions = set(actions)
        if len(unique_actions) > 1:
            print(f"✅ SUCESSO: A IA está viva! Variou entre {unique_actions}")
        else:
            print(f"⚠️ AVISO: A IA está teimosa (Apenas ação {list(unique_actions)[0]}).")
            
    except Exception as e:
        print(f"❌ Erro Crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_validation()