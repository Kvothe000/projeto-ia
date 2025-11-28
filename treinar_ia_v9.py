# Binance/treinar_ia_v9.py - GBT + REGULARIZAÇÃO + EARLY STOPPING
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import joblib

# Configurações
ARQUIVO_DADOS = "dataset_v8_atr.csv" # Usamos o dataset V8 que já está bom (ATR Manual)
ARQUIVO_MODELO = "modelo_ia_v6.pkl"  # Sobrescreve para o bot usar direto

def treinar():
    print("🧠 INICIANDO TREINO V9 (O Protocolo do Colega)...")
    
    try:
        df = pd.read_csv(ARQUIVO_DADOS)
    except:
        print("❌ Erro: Dataset não encontrado.")
        return

    print(f"📚 Dados Carregados: {len(df)} candles.")
    
    # Separa Features e Alvo
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 1. DIVISÃO TEMPORAL (Sem Shuffle - Respeita o Tempo)
    # 80% Passado (Treino) | 20% Futuro (Teste)
    corte = int(len(df) * 0.80)
    
    X_train = X.iloc[:corte]
    y_train = y.iloc[:corte]
    
    X_test = X.iloc[corte:]
    y_test = y.iloc[corte:]
    
    print(f"⏳ Treino: {len(X_train)} | Validação Futura: {len(X_test)}")

    # 2. BALANCEAMENTO DE CLASSES
    # Calcula pesos para a IA dar a mesma importância a Long/Short/Neutro
    sample_weights = class_weight.compute_sample_weight(
        class_weight='balanced', 
        y=y_train
    )

    # 3. MODELO V9 (HI-TECH)
    # Implementa as sugestões do colega: Regularização e Early Stopping
    print("🏋️ Treinando Gradient Boosting com Regularização L2...")
    
    modelo = HistGradientBoostingClassifier(
        learning_rate=0.05,      # Aprende devagar para não decorar (Overfitting)
        max_iter=2000,           # Muitas iterações permitidas...
        max_depth=8,             # Árvores menos profundas (Mais generalistas)
        min_samples_leaf=100,    # Exige confirmação forte de padrão
        l2_regularization=5.0,   # <--- A MÁGICA (Evita decorar ruído)
        early_stopping=True,     # <--- A MÁGICA (Para se parar de melhorar)
        validation_fraction=0.1, # Usa 10% do treino para saber quando parar
        n_iter_no_change=20,     # Se não melhorar em 20 rodadas, para.
        random_state=42
    )
    
    modelo.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 4. AVALIAÇÃO DE ELITE
    print("\n🔍 RESULTADO NO FUTURO (O Teste da Verdade):")
    probs = modelo.predict_proba(X_test)
    
    # Loop de Confiança
    melhor_precisao = 0
    
    for i, nome in enumerate(['NEUTRO', 'LONG', 'SHORT']):
        if nome == 'NEUTRO': continue
        
        print(f"\n--- ANÁLISE {nome} ---")
        for conf in [0.55, 0.60, 0.70]:
            mask = probs[:, i] > conf
            total_trades = mask.sum()
            
            if total_trades > 0:
                acertos = (y_test[mask] == i).sum()
                acc = acertos / total_trades
                print(f"Confiança > {conf*100:.0f}%: {total_trades} Trades -> {acc*100:.1f}% Acerto")
                if total_trades > 10 and acc > melhor_precisao:
                    melhor_precisao = acc
            else:
                print(f"Confiança > {conf*100:.0f}%: 0 Trades")

    # 5. SALVAR
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"\n💾 Modelo Salvo! (Precisão Base: {modelo.score(X_test, y_test)*100:.1f}%)")

if __name__ == "__main__":
    treinar()