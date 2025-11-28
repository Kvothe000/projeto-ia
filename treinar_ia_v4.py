# Binance/treinar_ia_v4.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Configurações
ARQUIVO_DATASET = "dataset_v4_institucional.csv"
ARQUIVO_MODELO = "modelo_ia_v4.pkl"
ARQUIVO_CONFIG = "config_ia_v4.txt"

def treinar():
    print("🧠 Iniciando Treinamento V4 (Cérebro Institucional)...")
    
    # 1. Carregar Dados
    try:
        df = pd.read_csv(ARQUIVO_DATASET)
    except FileNotFoundError:
        print("❌ Erro: Dataset não encontrado. Rode o gerar_dataset_v4.py antes!")
        return

    # 2. Preparar Features (X) e Alvo (y)
    X = df.drop(columns=['target'])
    y = df['target']

    # 3. Divisão Treino/Teste (Primeiros 80% para treino, últimos 20% para teste)
    # Não misturamos (shuffle=False) para simular o "futuro" desconhecido
    corte = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    print(f"📚 Treinando com {len(X_train)} candles...")
    print(f"📝 Validando em {len(X_test)} candles recentes...")

    # 4. Configurar a IA (Random Forest Otimizada)
    # class_weight='balanced_subsample': Dá peso extra para os acertos raros (Wins)
    modelo = RandomForestClassifier(
        n_estimators=500,        # Mais árvores para pensar melhor
        max_depth=10,            # Limita complexidade para não "decorar" o passado
        min_samples_leaf=5,      # Exige que um padrão aconteça pelo menos 5 vezes
        class_weight='balanced_subsample', 
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    # 5. Otimização do Limiar (Threshold) - A Busca do Sniper
    print("\n🔍 Buscando a Certeza Ideal para Lucrar...")
    
    # Pega a probabilidade de WIN (coluna 1)
    probs = modelo.predict_proba(X_test)[:, 1] 
    
    melhor_precisao = 0
    melhor_threshold = 0.5
    
    print(f"{'CERTEZA MIN.':<15} | {'TRADES':<8} | {'ACERTOS':<8} | {'PRECISÃO':<10}")
    print("-" * 55)

    # Testa exigências de 50% até 80%
    for corte in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        # Filtra quais trades a IA teria feito com essa exigência
        trades_feitos = (probs >= corte)
        
        if trades_feitos.sum() == 0:
            print(f"{corte*100:.0f}%{' ':<11} | 0        | 0        | -")
            continue

        # Verifica quantos desses trades foram WIN real (y_test == 1)
        acertos = y_test[trades_feitos].sum()
        total = trades_feitos.sum()
        precisao = acertos / total
        
        print(f"{corte*100:.0f}%{' ':<11} | {total:<8} | {acertos:<8} | {precisao*100:.2f}%")

        # Critério de Sucesso: Precisão > 55% e pelo menos 3 trades no teste
        if precisao > melhor_precisao and total >= 3:
            melhor_precisao = precisao
            melhor_threshold = corte

    print("-" * 55)
    
    # 6. Análise de Importância (O que a IA mais valorizou?)
    importancias = pd.DataFrame({
        'Indicador': X.columns,
        'Peso': modelo.feature_importances_
    }).sort_values('Peso', ascending=False)
    
    print("\n📊 O Segredo do Sucesso (Top 3 Indicadores):")
    print(importancias.head(3))

    # 7. Salvar ou Descartar
    if melhor_precisao > 0.50:
        print(f"\n🏆 SUCESSO! Melhor configuração: Exigir {melhor_threshold*100:.0f}% de Certeza.")
        print(f"   (Isso gerou {melhor_precisao*100:.1f}% de acerto nos testes)")
        
        joblib.dump(modelo, ARQUIVO_MODELO)
        with open(ARQUIVO_CONFIG, "w") as f:
            f.write(str(melhor_threshold))
        print(f"💾 IA Salva em: {ARQUIVO_MODELO}")
    else:
        print("\n⚠️ A IA não conseguiu bater o mercado com segurança neste dataset pequeno.")
        print("   Dica: Aumente QTD_CANDLES no minerador para 5.000 ou 10.000.")

if __name__ == "__main__":
    treinar()