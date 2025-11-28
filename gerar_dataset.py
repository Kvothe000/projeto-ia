# Binance/gerar_dataset.py
import pandas as pd
import numpy as np
from binance_connector import BinanceConnector
from indicators import Calculadora
import time

# --- CONFIGURAÇÕES ---
PAR = "BTCUSDT"
TIMEFRAME = "15m"
CANDLES_HISTORICO = 5000  # Quantidade de dados para a IA aprender (quanto mais, melhor)
ALVO_LUCRO = 0.005        # 0.5% de lucro (Meta)
ALVO_STOP = 0.005         # 0.5% de stop (Risco)
HORIZONTE_FUTURO = 10     # A IA deve prever o que acontece nos próximos 10 candles

def criar_alvo(df):
    """
    Cria a coluna 'TARGET' (O que a IA deve prever).
    1 = Deu Lucro (Bateu no TP antes do Stop)
    0 = Deu Prejuízo ou ficou no zero
    """
    targets = []
    
    print("🔮 Calculando o futuro para cada candle...")
    
    for i in range(len(df)):
        # Se não tiver dados futuros suficientes, ignora
        if i + HORIZONTE_FUTURO >= len(df):
            targets.append(0)
            continue
            
        preco_entrada = df.iloc[i]['close']
        tp_price = preco_entrada * (1 + ALVO_LUCRO)
        stop_price = preco_entrada * (1 - ALVO_STOP)
        
        resultado = 0 # Assume 0 (não deu bom)
        
        # Olha para o futuro (próximos X candles)
        for j in range(1, HORIZONTE_FUTURO + 1):
            futuro = df.iloc[i + j]
            
            # Tocou no Stop primeiro?
            if futuro['low'] <= stop_price:
                resultado = 0
                break # Game over
            
            # Tocou no Take Profit?
            if futuro['high'] >= tp_price:
                resultado = 1 # WIN!
                break # Vitória
                
        targets.append(resultado)
        
    return targets

def main():
    print(f"🚀 Iniciando coleta de dados para {PAR}...")
    connector = BinanceConnector()
    
    # 1. Baixar Histórico Gigante
    # O método buscar_candles do seu connector baixa 500 por padrão. 
    # Vamos ter que fazer um loop ou aceitar 500/1000 para teste inicial.
    # Nota: Para produção, precisaríamos ajustar o connector para baixar mais, 
    # mas para esse teste vamos usar o máximo que ele permitir.
    print("📥 Baixando candles da Binance...")
    df = connector.buscar_candles(PAR, TIMEFRAME) 
    
    if df is None:
        print("❌ Erro ao baixar dados.")
        return

    print(f"✅ {len(df)} candles baixados.")

    # 2. Calcular Indicadores (O que a IA vai 'ver')
    print("🧮 Calculando indicadores técnicos...")
    df = Calculadora.adicionar_todos(df)
    
    # Remove colunas vazias geradas pelos indicadores iniciais
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3. Criar o Gabarito (Target)
    df['target'] = criar_alvo(df)

    # 4. Limpeza Final
    # Vamos selecionar apenas colunas numéricas relevantes para a IA
    colunas_ia = [
        'open', 'high', 'low', 'close', 'volume', # Preço puro
        'RSI_14', 'ADX_14', 'ATRr_14',            # Indicadores de força/volatilidade
        'BBL_20_2.0', 'BBU_20_2.0',               # Bollinger
        'EMA_9', 'EMA_21', 'EMA_200',             # Médias
        'target'                                  # O QUE QUEREMOS PREVER
    ]
    
    # Filtra só o que existe no DF (para evitar erro se algum indicador falhar)
    cols_existentes = [c for c in colunas_ia if c in df.columns]
    df_final = df[cols_existentes]
    
    # Remove as últimas linhas onde não sabemos o futuro ainda
    df_final = df_final.iloc[:-HORIZONTE_FUTURO]

    # 5. Salvar CSV
    nome_arquivo = "dataset_treino_v1.csv"
    df_final.to_csv(nome_arquivo, index=False)
    
    print(f"\n✨ SUCESSO! Dataset gerado: {nome_arquivo}")
    print(f"📊 Total de exemplos para treino: {len(df_final)}")
    print(f"📈 Taxa de Win teórica no período: {(df_final['target'].mean() * 100):.2f}%")
    print("👉 Próximo passo: Treinar a IA com este arquivo.")

if __name__ == "__main__":
    main()