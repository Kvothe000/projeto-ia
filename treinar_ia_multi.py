# Binance/treinar_ia_multi.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

ARQUIVO = "dataset_universe.csv"
ARQUIVO_MODELO = "modelo_ia_v5.pkl" # Sobrescreve o antigo V5

def treinar():
    print("🧠 Treinando CÉREBRO GENERALISTA (10 Moedas)...")
    try:
        df = pd.read_csv(ARQUIVO)
    except:
        print("❌ Arquivo não encontrado!")
        return

    print(f"📚 Carregando {len(df)} exemplos de batalha...")
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Divisão (Como embaralhamos no gerador, podemos pegar random)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Modelo Robusto (Mais árvores para aguentar a variedade)
    modelo = RandomForestClassifier(
        n_estimators=800,        # Mais poder de processamento
        max_depth=15,            # Um pouco mais profundo para entender nuances
        min_samples_leaf=10,     # Exige padrões fortes (evita decorar ruído)
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    print("🏋️ Iniciando treino pesado (isso pode demorar uns minutos)...")
    modelo.fit(X_train, y_train)
    
    # Avaliação
    print("\n🔍 RESULTADO DA PROVA FINAL:")
    probs = modelo.predict_proba(X_test)
    
    # Teste de Longs
    threshold = 0.55 # Começamos exigindo 55%
    longs = (probs[:, 1] > threshold)
    if longs.sum() > 0:
        acc = (y_test[longs] == 1).mean()
        print(f"📈 COMPRA (>{threshold*100}%): {longs.sum()} Trades -> {acc*100:.1f}% Precisão")
    
    # Teste de Shorts
    shorts = (probs[:, 2] > threshold)
    if shorts.sum() > 0:
        acc = (y_test[shorts] == 2).mean()
        print(f"📉 VENDA  (>{threshold*100}%): {shorts.sum()} Trades -> {acc*100:.1f}% Precisão")

    # Salvar
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"\n💾 Cérebro Atualizado! Salvo em: {ARQUIVO_MODELO}")
    print("👉 O seu bot 'main_final.py' já vai usar este novo cérebro automaticamente.")

if __name__ == "__main__":
    treinar()