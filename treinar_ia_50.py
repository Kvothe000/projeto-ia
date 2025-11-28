# Binance/treinar_ia_50.py (UPGRADE GRADIENT BOOSTING)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier # <--- O NOVO MOTOR
from sklearn.utils import class_weight
import joblib

ARQUIVO = "dataset_50_coins_norm.csv"
ARQUIVO_MODELO = "modelo_ia_v5.pkl"

def treinar():
    print("🧠 Carregando Cérebro GBT (50 Moedas)...")
    try:
        df = pd.read_csv(ARQUIVO)
    except:
        print("❌ Rode o gerar_dataset_50.py primeiro!")
        return

    print(f"📚 Estudando {len(df)} candles...")
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Divisão
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Calcular pesos (para equilibrar classes desbalanceadas)
    # Gradient Boosting não tem parâmetro class_weight automático no sklearn,
    # então usamos sample_weight na hora do fit
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    print("🏋️ Treinando Gradient Boosting (Isso é rápido!)...")
    
    # HistGradientBoostingClassifier é inspirado no LightGBM (Muito rápido e preciso)
    modelo = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=300,           # Equivalente a n_estimators
        max_depth=12,
        min_samples_leaf=50,    # Mais conservador
        random_state=42,
        early_stopping=True     # Para de treinar se não melhorar (evita overfitting)
    )
    
    modelo.fit(X_train, y_train, sample_weight=classes_weights)
    
    print("\n🔍 Teste de Precisão:")
    probs = modelo.predict_proba(X_test)
    
    # Relatório Detalhado
    for i, nome in enumerate(['NEUTRO', 'LONG', 'SHORT']):
        if nome == 'NEUTRO': continue
        
        # Testa vários níveis de confiança
        for threshold in [0.55, 0.60, 0.70]:
            mask = probs[:, i] > threshold
            total = mask.sum()
            if total > 0:
                acertos = (y_test[mask] == i).sum()
                acc = acertos / total
                print(f"{nome} (> {threshold*100:.0f}%): {total} Trades -> {acc*100:.1f}% Acerto")
            else:
                print(f"{nome} (> {threshold*100:.0f}%): 0 Trades")

    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"\n💾 Cérebro Salvo: {ARQUIVO_MODELO}")

if __name__ == "__main__":
    treinar()