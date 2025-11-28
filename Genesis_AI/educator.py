# Genesis_AI/educator.py
from brain import GenesisBrain
import os

# Caminho para os dados ricos que já mineramos na pasta Binance
DATASET_PATH = "../Binance/dataset_v11_fusion.csv"
MODEL_PATH = "cerebros/genesis_alpha"

def iniciar_educacao():
    print("🎓 BEM-VINDO AO PROJETO GÊNESIS")
    print("===============================")
    
    # 1. Instancia a IA
    genesis = GenesisBrain(DATASET_PATH, MODEL_PATH)
    
    # 2. Tenta carregar conhecimento prévio ou nasce
    if not genesis.carregar():
        genesis.nascer()
    
    # 3. Loop de Educação Contínua
    # A IA vai viver "1 milhão de candles" repetidamente para aprender
    ciclos = 5
    passos_por_ciclo = 50000 
    
    for i in range(ciclos):
        print(f"\n🔄 Ciclo de Evolução {i+1}/{ciclos}")
        genesis.treinar(passos=passos_por_ciclo)
        print(f"✅ Ciclo {i+1} concluído. O cérebro está a evoluir.")

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERRO: Não encontrei {DATASET_PATH}")
        print("   Por favor, vá na pasta 'Binance' e rode 'python gerar_dataset_v11_fusion.py' primeiro.")
    else:
        iniciar_educacao()