import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- CONFIGURAÇÕES ---
ARQUIVO_DATASET = "dataset_treino_v1.csv"
ARQUIVO_MODELO = "modelo_ia_v1.pkl"

def treinar():
    print("🧠 Carregando os dados para estudo...")
    try:
        df = pd.read_csv(ARQUIVO_DATASET)
    except FileNotFoundError:
        print("❌ Erro: Arquivo 'dataset_treino_v1.csv' não encontrado.")
        print("   Rode o script 'gerar_dataset.py' primeiro!")
        return

    # 1. Separar o que é "Dica" (X) do que é "Resposta" (y)
    # Removemos a coluna 'target' e datas/preços que não ajudam na previsão relativa
    # (A IA não deve decorar o preço do Bitcoin em 2022, mas sim o comportamento dos indicadores)
    colunas_proibidas = ['target', 'timestamp', 'close_time', 'open', 'high', 'low', 'close', 'ignore']
    
    # Filtra colunas que realmente existem no dataset
    cols_drop = [c for c in colunas_proibidas if c in df.columns]
    
    X = df.drop(columns=cols_drop) # Tudo que sobrou são indicadores (RSI, ADX, EMAs...)
    y = df['target']               # 1 = Lucro, 0 = Prejuízo/Nada

    print(f"📊 Colunas usadas para treino: {list(X.columns)}")

    # 2. Dividir em "Estudo" (Treino) e "Prova" (Teste)
    # 80% para estudar, 20% para testar se aprendeu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    print(f"📚 Estudando {len(X_train)} exemplos...")
    print(f"📝 Deixando {len(X_test)} exemplos para a prova final...")

    # 3. Criar e Treinar o Modelo (Random Forest)
    # n_estimators=100 -> Cria 100 "árvores de decisão" e faz uma votação
    modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X_train, y_train)

    # 4. Avaliar o Desempenho
    print("\n🎓 Resultado da Prova Final (Nos dados de teste):")
    previsoes = modelo.predict(X_test)
    
    # Relatório completo
    print(classification_report(y_test, previsoes, target_names=['Neutro/Loss', 'WIN (Lucro)']))
    
    # Matriz de Confusão (Importante!)
    cm = confusion_matrix(y_test, previsoes)
    tn, fp, fn, tp = cm.ravel()
    
    print("-" * 30)
    print(f"✅ ACERTOS DE WIN (Verdadeiros Positivos): {tp}")
    print(f"❌ ALARMES FALSOS (Falsos Positivos): {fp}")
    print(f"⚠️  OPORTUNIDADES PERDIDAS (Falsos Negativos): {fn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"\n🎯 PRECISÃO DA IA (Quando ela diz 'COMPRA', ela acerta?): {precision*100:.2f}%")

    # 5. Salvar o Cérebro
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"\n💾 Modelo salvo com sucesso em: {ARQUIVO_MODELO}")
    print("🚀 Agora você pode usar este arquivo no seu bot!")

if __name__ == "__main__":
    treinar()