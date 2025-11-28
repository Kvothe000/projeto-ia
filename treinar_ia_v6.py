# Binance/treinar_ia_v6.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import class_weight
import joblib

ARQUIVO = "dataset_v6_context.csv"
ARQUIVO_MODELO = "modelo_ia_v6.pkl"

def treinar():
    print("🧠 Treinando IA V6 (Com Consciência de Mercado)...")
    try:
        df = pd.read_csv(ARQUIVO)
    except:
        print("❌ Rode o gerar_dataset_v6.py primeiro!")
        return

    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

    print("🏋️ Treinando Gradient Boosting...")
    modelo = HistGradientBoostingClassifier(
        learning_rate=0.05,     
        max_iter=500,           
        max_depth=10,
        min_samples_leaf=50,
        random_state=42,
        early_stopping=True
    )
    
    modelo.fit(X_train, y_train, sample_weight=classes_weights)
    
    print("\n🔍 Teste de Precisão V6 (Esperado > 50%):")
    probs = modelo.predict_proba(X_test)
    
    for i, nome in enumerate(['NEUTRO', 'LONG', 'SHORT']):
        if nome == 'NEUTRO': continue
        threshold = 0.55
        mask = probs[:, i] > threshold
        if mask.sum() > 0:
            acc = (y_test[mask] == i).mean()
            print(f"{nome} (> {threshold*100:.0f}%): {mask.sum()} Trades -> {acc*100:.1f}% Acerto")

    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"\n💾 Cérebro V6 Salvo: {ARQUIVO_MODELO}")

if __name__ == "__main__":
    treinar()