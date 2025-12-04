# Genesis_AI/test_genesis_2025.py (CORREÇÃO DE SHAPE DINÂMICA)
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fixed_trading_env import RealisticTradingEnv
import matplotlib.pyplot as plt
import os

# CONFIG
MODELO_NOME = "genesis_wld_v2"
NOME_ARQUIVO_DADOS = "dataset_2025.csv"
NOME_ARQUIVO_TREINO = "dataset_wld_clean.csv" # Referência para pegar as colunas certas
WINDOW_SIZE = 30

def run_test():
    print("⚖️ EXECUTANDO TESTE CEGO (ÚLTIMOS 25% DE 2025)...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Encontra o Modelo
    modelo_path = os.path.join(base_dir, "cerebros", MODELO_NOME)
    if not os.path.exists(modelo_path + ".zip") and not os.path.exists(modelo_path):
        print(f"❌ Modelo não encontrado: {modelo_path}")
        return
    
    print(f"🧠 Carregando modelo...")
    model = PPO.load(modelo_path)

    # 2. Encontra Dataset de Referência (Treino) para copiar a estrutura
    ref_path = None
    possible_refs = [
        os.path.join(base_dir, "..", "Binance", NOME_ARQUIVO_TREINO),
        os.path.join(base_dir, NOME_ARQUIVO_TREINO),
        NOME_ARQUIVO_TREINO
    ]
    for p in possible_refs:
        if os.path.exists(p): ref_path = p; break
        
    if not ref_path:
        print("❌ Erro: Dataset de treino original não encontrado para referência de colunas.")
        return
        
    print(f"📋 Lendo esquema do treino: {ref_path}")
    df_ref = pd.read_csv(ref_path)
    df_ref_num = df_ref.select_dtypes(include=[np.number])
    cols_drop_ref = ['target', 'timestamp', 'close']
    df_ref_clean = df_ref_num.drop(columns=[c for c in cols_drop_ref if c in df_ref_num.columns])
    COLS_EXPECTED = df_ref_clean.columns.tolist()
    print(f"   ✅ O modelo espera {len(COLS_EXPECTED)} features: {COLS_EXPECTED}")

    # 3. Encontra Dataset de Teste (2025)
    dados_path = None
    possible_datas = [
        os.path.join(base_dir, "..", "Binance", NOME_ARQUIVO_DADOS),
        os.path.join(base_dir, NOME_ARQUIVO_DADOS),
        NOME_ARQUIVO_DADOS
    ]
    for p in possible_datas:
        if os.path.exists(p): dados_path = p; break
            
    if not dados_path:
        print(f"❌ Erro: Dataset '{NOME_ARQUIVO_DADOS}' não encontrado.")
        return

    print(f"📚 Carregando dados de teste: {dados_path}")
    df = pd.read_csv(dados_path)
    
    # --- RECUPERA OS DADOS DE TESTE (SPLIT 75/25) ---
    corte = int(len(df) * 0.75)
    df_treino_stats = df.iloc[:corte] # Usado apenas para calcular média/desvio
    df_teste = df.iloc[corte:].reset_index(drop=True) # Onde vamos operar
    
    # Prepara Preço Real (Para o ambiente calcular lucro)
    price_test = df_teste['close'].values
    
    # --- ALINHAMENTO DE COLUNAS (A MÁGICA) ---
    # Força o dataset de teste a ter as MESMAS colunas do treino
    # Se faltar alguma, preenche com 0. Se tiver extra, ignora.
    
    # 1. Normalização (Calcula stats nas colunas certas)
    # Filtra colunas no DF de estatística
    for col in COLS_EXPECTED:
        if col not in df_treino_stats.columns: df_treino_stats[col] = 0
    df_stats_clean = df_treino_stats[COLS_EXPECTED]
    
    mean_train = df_stats_clean.mean()
    std_train = df_stats_clean.std()
    
    # 2. Prepara o DF de Teste
    for col in COLS_EXPECTED:
        if col not in df_teste.columns: df_teste[col] = 0 # Preenche buracos
    
    df_test_feat = df_teste[COLS_EXPECTED].copy() # Seleciona apenas as colunas certas na ordem certa
    
    # 3. Aplica Normalização
    df_test_norm = (df_test_feat - mean_train) / std_train
    df_test_norm = df_test_norm.fillna(0).clip(-5, 5)
    
    print(f"📉 Simulando em {len(df_test_norm)} candles desconhecidos (Shape: {df_test_norm.shape})...")
    
    # Ambiente
    env = DummyVecEnv([lambda: RealisticTradingEnv(
        df_test_norm, 
        price_test, 
        initial_balance=10000, 
        lookback_window=WINDOW_SIZE
    )])
    
    obs = env.reset()
    done = False
    equity = [10000]
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        equity.append(info[0]['net_worth'])
        
    # Relatório
    saldo_final = equity[-1]
    lucro_pct = (saldo_final - 10000) / 10000 * 100
    
    # Drawdown
    peak = 10000
    max_dd = 0
    for v in equity:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd: max_dd = dd
        
    print("="*40)
    print(f"📅 RESULTADO TESTE CEGO (25% FINAL)")
    print("="*40)
    print(f"💰 Saldo Final:   ${saldo_final:,.2f}")
    print(f"📈 Lucro Total:   {lucro_pct:.2f}%")
    print(f"📉 Drawdown Máx:  {max_dd:.2f}%")
    
    if lucro_pct > 0: print("✅ APROVADO: A IA previu o futuro corretamente.")
    else: print("❌ REPROVADO: A IA falhou no teste cego.")

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(equity, label="Patrimônio (Teste Cego)", color='purple')
        plt.axhline(y=10000, color='r', linestyle='--', label="Inicial")
        plt.title(f"Genesis 2025 - Out-of-Sample Test (Lucro: {lucro_pct:.2f}%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("genesis_2025_result.png")
        print("📉 Gráfico salvo: genesis_2025_result.png")
    except: pass

if __name__ == "__main__":
    run_test()