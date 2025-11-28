# Binance/treinar_ia_v3.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

ARQUIVO = "dataset_v3_turbo.csv"

def treinar():
    print("🏋️ Iniciando Treino V3 (Focado em Alta Precisão)...")
    try:
        df = pd.read_csv(ARQUIVO)
    except:
        print("❌ Rode o gerar_dataset_v3.py primeiro!")
        return

    X = df.drop(columns=['target'])
    y = df['target']
    
    # Vamos usar os últimos 20% para teste (Simulando o mercado recente)
    corte = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    # Modelo mais robusto (menos árvores profundas para evitar decorar ruído)
    modelo = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,  # Exige que pelo menos 5 exemplos confirmem uma regra
        max_depth=10,        # Não deixa a árvore ficar muito complexa
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    modelo.fit(X_train, y_train)
    
    # --- A MÁGICA DO THRESHOLD (LIMIAR DE CONFIANÇA) ---
    print("\n🔍 Analisando Probabilidades...")
    
    # A IA retorna duas colunas: [Probabilidade de Perder, Probabilidade de Ganhar]
    # Nós queremos a coluna 1 (Ganhar)
    probs = modelo.predict_proba(X_test)[:, 1]
    
    # Vamos testar vários níveis de exigência
    for exigencia in [0.50, 0.60, 0.70, 0.75, 0.80]:
        # Só consideramos COMPRA se a certeza for maior que a exigência
        trades_feitos = (probs >= exigencia).astype(int)
        
        # Comparamos com a realidade
        cm = confusion_matrix(y_test, trades_feitos)
        tn, fp, fn, tp = cm.ravel()
        
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
        total_trades = tp + fp
        
        print(f"\n--- Exigindo {exigencia*100}% de Certeza ---")
        print(f"🛒 Trades Realizados: {total_trades}")
        print(f"✅ Acertos (Lucro): {tp}")
        print(f"❌ Erros (Prejuízo): {fp}")
        print(f"🎯 PRECISÃO: {precisao*100:.2f}%")
        
        if precisao > 0.60 and total_trades > 10:
            print("🚀 ACHEI! Esse nível de exigência dá lucro!")
            # Salvar modelo e metadados
            joblib.dump(modelo, "modelo_ia_v3.pkl")
            # Salvar a exigência ideal num txt para o bot ler depois
            with open("config_ia.txt", "w") as f:
                f.write(str(exigencia))

if __name__ == "__main__":
    treinar()