# Binance/treinar_ia_v8.py (SAFE MODE)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
import joblib

ARQUIVO = "dataset_v8_atr.csv"
ARQUIVO_MODELO = "modelo_ia_v6.pkl"

def treinar():
    print("🧠 Treinando IA V8 (Random Forest + ATR Target)...")
    try:
        df = pd.read_csv(ARQUIVO)
    except:
        print("❌ Arquivo não encontrado!")
        return

    # Validação de Classes
    counts = df['target'].value_counts()
    print(f"📊 Distribuição:\n{counts}")
    
    if len(counts) < 2:
        print("❌ ERRO: O Dataset só tem uma classe (provavelmente só zeros).")
        print("   A IA não tem o que aprender. Rode o 'gerar_dataset_v8.py' novamente.")
        return

    X = df.drop(columns=['target'])
    y = df['target']
    
    corte = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]
    
    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

    print("🏋️ Treinando Random Forest...")
    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=50,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    modelo.fit(X_train, y_train, sample_weight=weights)
    
    print("\n🔍 Validação Futura:")
    probs = modelo.predict_proba(X_test)
    
    # Verifica se o modelo aprendeu as 3 classes
    classes_aprendidas = modelo.classes_
    print(f"Classes aprendidas: {classes_aprendidas}")

    for i, classe in enumerate(classes_aprendidas):
        nome = "NEUTRO"
        if classe == 1: nome = "LONG"
        if classe == 2: nome = "SHORT"
        if nome == "NEUTRO": continue
        
        # Se a classe existe no modelo, testa
        # Cuidado: probs tem colunas na ordem de 'classes_aprendidas'
        threshold = 0.52
        mask = probs[:, i] > threshold
        if mask.sum() > 0:
            acc = (y_test[mask] == classe).mean()
            print(f"{nome} (> {threshold*100:.0f}%): {mask.sum()} Trades -> {acc*100:.1f}% Acerto")
        else:
            print(f"{nome}: 0 Trades")

    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"\n💾 Cérebro Salvo: {ARQUIVO_MODELO}")

if __name__ == "__main__":
    treinar()