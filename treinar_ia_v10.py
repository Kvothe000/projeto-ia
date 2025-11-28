# Binance/treinar_ia_v10.py - VALIDAÇÃO RIGOROSA
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

ARQUIVO = "dataset_v10_seguro.csv"
ARQUIVO_MODELO = "modelo_ia_v10.pkl"

def treinar():
    print("🧠 INICIANDO TREINO V10...")
    try:
        df = pd.read_csv(ARQUIVO)
    except:
        print("❌ Dataset não encontrado!")
        return

    # Se tiver muitos neutros (0), podemos filtrar ou usar pesos
    # Vamos manter tudo para a IA aprender a "não fazer nada"
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # DIVISÃO TEMPORAL REAL (Últimos 20% para Teste Final)
    corte = int(len(df) * 0.8)
    X_train_full = X.iloc[:corte]
    y_train_full = y.iloc[:corte]
    X_test = X.iloc[corte:]
    y_test = y.iloc[corte:]
    
    print(f"📚 Dados: {len(df)} | Treino: {len(X_train_full)} | Teste Futuro: {len(X_test)}")

    # VALIDAÇÃO CRUZADA TEMPORAL (Walk-Forward no Treino)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    print("\n⏳ Validando consistência do modelo (Cross-Validation)...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full)):
        X_t, X_v = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_t, y_v = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=6, max_iter=200, 
            l2_regularization=1.0, random_state=42
        )
        model.fit(X_t, y_t)
        acc = model.score(X_v, y_v)
        scores.append(acc)
        print(f"   Fold {fold+1}: Acurácia {acc*100:.1f}%")
        
    print(f"📊 Média Validação: {np.mean(scores)*100:.1f}%")

    # TREINO FINAL E TESTE NO FUTURO
    print("\n🚀 Treinando modelo final...")
    modelo_final = HistGradientBoostingClassifier(
        learning_rate=0.03, max_depth=8, max_iter=500, 
        l2_regularization=2.0, early_stopping=True, random_state=42
    )
    modelo_final.fit(X_train_full, y_train_full)
    
    probs = modelo_final.predict_proba(X_test)
    
    print("\n🔍 TESTE DE FOGO (FUTURO):")
    for i, nome in enumerate(['NEUTRO', 'LONG', 'SHORT']):
        if nome == 'NEUTRO': continue
        
        for conf in [0.55, 0.60]:
            mask = probs[:, i] > conf
            total = mask.sum()
            if total > 0:
                acc = (y_test[mask] == i).mean()
                print(f"{nome} (> {conf*100:.0f}%): {total} Trades -> {acc*100:.1f}% Acerto")
            else:
                print(f"{nome} (> {conf*100:.0f}%): 0 Trades")

    joblib.dump(modelo_final, ARQUIVO_MODELO)
    print(f"\n💾 Modelo V10 Salvo!")

if __name__ == "__main__":
    treinar()
    