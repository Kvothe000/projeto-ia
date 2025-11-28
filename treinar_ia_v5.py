# Binance/treinar_ia_v5.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

ARQUIVO = "dataset_v5_wld.csv"

def treinar():
    print("🧠 Treinando IA V5 (Long & Short)...")
    try:
        df = pd.read_csv(ARQUIVO)
    except:
        print("❌ Rode o gerar_dataset_v5.py primeiro!")
        return

    X = df.drop(columns=['target'])
    y = df['target']
    
    corte = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    print(f"📚 Estudando {len(X_train)} candles da WLD...")

    # Modelo Multi-Classe
    modelo = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=4,
        class_weight='balanced', # Importante para equilibrar Long/Short/Neutro
        random_state=42,
        n_jobs=-1
    )
    
    modelo.fit(X_train, y_train)
    
    # Avaliação
    print("\n🔍 Testando Precisão nos dados recentes...")
    # Probabilidades: [Neutro%, Long%, Short%]
    probs = modelo.predict_proba(X_test)
    
    # Vamos testar a confiança para LONG (Classe 1)
    print("\n--- ANÁLISE DE LONGS (COMPRA) ---")
    threshold = 0.60
    long_probs = probs[:, 1]
    trades = (long_probs > threshold)
    if trades.sum() > 0:
        acertos = y_test[trades] == 1
        print(f"Com {threshold*100}% de certeza: {trades.sum()} Trades -> {acertos.mean()*100:.1f}% Precisão")
    else:
        print("Nenhum trade Long com essa confiança.")

    # Vamos testar a confiança para SHORT (Classe 2)
    print("\n--- ANÁLISE DE SHORTS (VENDA) ---")
    short_probs = probs[:, 2]
    trades = (short_probs > threshold)
    if trades.sum() > 0:
        acertos = y_test[trades] == 2
        print(f"Com {threshold*100}% de certeza: {trades.sum()} Trades -> {acertos.mean()*100:.1f}% Precisão")
    else:
        print("Nenhum trade Short com essa confiança.")
        
    joblib.dump(modelo, "modelo_ia_v5.pkl")
    print("\n💾 Modelo V5 salvo!")

if __name__ == "__main__":
    treinar()