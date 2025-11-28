import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

ARQUIVO_DATASET = "dataset_grande.csv" # Usando o novo arquivo maior

def treinar():
    print("🧠 Carregando dataset GIGANTE...")
    try:
        df = pd.read_csv(ARQUIVO_DATASET)
    except:
        print("❌ Rode o gerar_dataset_v2.py primeiro!")
        return

    # Limpeza básica
    cols_drop = ['target', 'timestamp', 'close_time', 'open', 'high', 'low', 'close', 'ignore']
    X = df.drop(columns=[c for c in cols_drop if c in df.columns])
    y = df['target']

    # Divisão
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"📚 Treinando com {len(X_train)} exemplos...")

    # --- A MÁGICA ACONTECE AQUI ---
    # class_weight='balanced': Dá mais peso para os acertos raros (Wins)
    # n_estimators=200: Mais árvores para pensar melhor
    modelo = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        class_weight='balanced', 
        random_state=42
    )
    
    modelo.fit(X_train, y_train)

    # Avaliação
    print("\n🎓 Resultado V2:")
    previsoes = modelo.predict(X_test)
    
    print(classification_report(y_test, previsoes, target_names=['Neutro', 'WIN']))
    
    cm = confusion_matrix(y_test, previsoes)
    tn, fp, fn, tp = cm.ravel()
    
    print("-" * 30)
    print(f"✅ ACERTOS REAIS (Lucro): {tp}")
    print(f"❌ FALSOS POSITIVOS (Prejuízo): {fp}")
    print("-" * 30)
    
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"🎯 PRECISÃO REAL: {precisao*100:.2f}%")
    
    if precisao > 0.55:
        print("🚀 EXCELENTE! A IA está melhor que aleatória. Pode salvar.")
        joblib.dump(modelo, "modelo_ia_v2.pkl")
    else:
        print("⚠️ Cuidado. A precisão está baixa. Talvez precise de mais indicadores.")

if __name__ == "__main__":
    treinar()