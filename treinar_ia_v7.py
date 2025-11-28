# Binance/treinar_ia_v7.py
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import class_weight
import joblib

ARQUIVO = "dataset_v7_timeseries.csv"
ARQUIVO_MODELO = "modelo_ia_v7.pkl"

def treinar():
    print("🧠 Treinando IA V7 (Validação Temporal Rigorosa)...")
    try:
        df = pd.read_csv(ARQUIVO)
    except:
        print("❌ Rode o gerar_dataset_v7.py primeiro!")
        return

    print(f"📚 Dados carregados: {len(df)} candles ordenados no tempo.")
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # DIVISÃO TEMPORAL (SEM SHUFFLE)
    # A IA aprende com o passado para prever o futuro
    corte = int(len(df) * 0.80)
    
    X_train = X.iloc[:corte]
    y_train = y.iloc[:corte]
    
    X_test = X.iloc[corte:]
    y_test = y.iloc[corte:]
    
    print(f"⏳ Treino: {len(X_train)} | Teste (Futuro): {len(X_test)}")
    
    # Pesos para balancear (Shorts e Longs são raros)
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

    print("🏋️ Treinando HistGradientBoosting...")
    modelo = HistGradientBoostingClassifier(
        learning_rate=0.03,     # Mais lento e preciso
        max_iter=1000,          # Mais iterações
        max_depth=12,
        min_samples_leaf=100,   # Exige padrões muito sólidos
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1 # Usa 10% do treino para validar parada
    )
    
    modelo.fit(X_train, y_train, sample_weight=classes_weights)
    
    print("\n🔍 Teste de Precisão no Futuro (O Teste de Fogo):")
    probs = modelo.predict_proba(X_test)
    
    # Relatório
    for i, nome in enumerate(['NEUTRO', 'LONG', 'SHORT']):
        if nome == 'NEUTRO': continue
        
        # Testamos níveis de confiança
        for conf in [0.55, 0.60, 0.70]:
            mask = probs[:, i] > conf
            total = mask.sum()
            if total > 0:
                acertos = (y_test[mask] == i).sum()
                acc = acertos / total
                print(f"{nome} (> {conf*100:.0f}%): {total} Trades -> {acc*100:.1f}% Acerto")
            else:
                print(f"{nome} (> {conf*100:.0f}%): 0 Trades")

    # Feature Importance (Permutation - aproximado para GBT)
    # Como o sklearn não dá feature_importances_ direto para GBT, pulamos o print
    # Mas sabemos que as novas features estão lá.

    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"\n💾 Cérebro V7 Salvo: {ARQUIVO_MODELO}")

if __name__ == "__main__":
    treinar()