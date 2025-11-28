# Binance/backtest_walk_forward.py - A PROVA DA VERDADE
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import os

# Configuração
ARQUIVO_DADOS = "dataset_v8_atr.csv"
JANELA_TREINO = 20000  # Treina com ~7 meses de dados (considerando 50 moedas misturadas)
JANELA_TESTE = 5000    # Opera por ~1.5 mês antes de re-treinar
MIN_CONFIANCA = 0.55   # Só entra se tiver 55% certeza

def backtest_deslizante():
    print("⏳ Iniciando Walk-Forward Analysis (Re-treino Contínuo)...")
    
    if not os.path.exists(ARQUIVO_DADOS):
        print("❌ Arquivo de dados não encontrado.")
        return

    df = pd.read_csv(ARQUIVO_DADOS)
    total_linhas = len(df)
    print(f"📚 Total de Dados: {total_linhas} candles.")
    
    # Preparação
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Loop Deslizante
    inicio = 0
    resultados = []
    
    while inicio + JANELA_TREINO + JANELA_TESTE < total_linhas:
        fim_treino = inicio + JANELA_TREINO
        fim_teste = fim_treino + JANELA_TESTE
        
        print(f"\n🔄 Ciclo: Linhas {inicio} até {fim_teste}...")
        
        # 1. Treino (Passado)
        X_train = X.iloc[inicio:fim_treino]
        y_train = y.iloc[inicio:fim_treino]
        
        # Balanceamento
        weights = class_weight.compute_sample_weight('balanced', y_train)
        
        modelo = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=8, max_iter=200, 
            min_samples_leaf=50, l2_regularization=5.0, random_state=42
        )
        modelo.fit(X_train, y_train, sample_weight=weights)
        
        # 2. Teste (Futuro Imediato)
        X_test = X.iloc[fim_treino:fim_teste]
        y_test = y.iloc[fim_treino:fim_teste]
        
        probs = modelo.predict_proba(X_test)
        
        # Simulação de Trades
        trades_ciclo = 0
        acertos_ciclo = 0
        
        # Analisa Longs (Classe 1) e Shorts (Classe 2)
        for i in [1, 2]: 
            mask = probs[:, i] > MIN_CONFIANCA
            if mask.sum() > 0:
                trades_ciclo += mask.sum()
                acertos = (y_test[mask] == i).sum()
                acertos_ciclo += acertos
        
        precisao = acertos_ciclo / trades_ciclo if trades_ciclo > 0 else 0
        resultados.append({
            'inicio': inicio,
            'trades': trades_ciclo,
            'precisao': precisao
        })
        
        print(f"   👉 Resultado: {trades_ciclo} Trades | Precisão: {precisao*100:.1f}%")
        
        # Desliza a janela (Avança o tamanho do teste)
        inicio += JANELA_TESTE

    # Relatório Final
    print("\n" + "="*40)
    print("📊 RESULTADO FINAL (MÉDIA DE TODOS OS CICLOS)")
    print("="*40)
    
    df_res = pd.DataFrame(resultados)
    total_trades = df_res['trades'].sum()
    precisao_media = (df_res['trades'] * df_res['precisao']).sum() / total_trades if total_trades > 0 else 0
    
    print(f"Total de Trades Simulados: {total_trades}")
    print(f"Precisão Média Global: {precisao_media*100:.2f}%")
    
    # Análise de Lucro Simplificada (Risco 1 : Retorno 1.5)
    # Lucro Esperado = (Acerto * 1.5) - (Erro * 1.0)
    ev = (precisao_media * 1.5) - ((1 - precisao_media) * 1.0)
    
    print(f"Expectativa Matemática (EV): {ev:.2f}R")
    if ev > 0:
        print("✅ ESTRATÉGIA LUCRATIVA COM RE-TREINO!")
        print("   Recomendação: Implementar sistema de Auto-Retrain.")
    else:
        print("❌ ESTRATÉGIA PREJUÍZO MESMO COM RE-TREINO.")
        print("   Recomendação: Precisamos de novos indicadores (Sentimento/Funding).")

if __name__ == "__main__":
    backtest_deslizante()