import pandas as pd
from binance.client import Client
import config
from indicators import Calculadora
import time

# --- CONFIGURAÇÕES ---
PAR = "BTCUSDT"
TIMEFRAME = Client.KLINE_INTERVAL_15MINUTE
QTD_CANDLES = 15000  # Vamos buscar MUITO mais dados
ALVO_LUCRO = 0.008   # 0.8%
ALVO_STOP = 0.005    # 0.5%
FUTURO_VISAO = 12    # Olha 12 candles para frente (3 horas)

def baixar_historico_profundo():
    print(f"🚀 Iniciando mineração profunda: {QTD_CANDLES} candles...")
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    
    # Busca em chunks para não dar erro na Binance
    klines = client.get_historical_klines(
        symbol=PAR, 
        interval=TIMEFRAME, 
        start_str=f"{int(QTD_CANDLES * 15 / 60 / 24) + 10} days ago UTC" 
    )
    
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tbba', 'tbqa', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    
    # Converter números
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    print(f"✅ Download concluído: {len(df)} linhas baixadas.")
    return df

def criar_alvo(df):
    targets = []
    print("🔮 Calculando o futuro (Gabarito)...")
    
    for i in range(len(df)):
        if i + FUTURO_VISAO >= len(df):
            targets.append(0)
            continue
            
        preco_entrada = df.iloc[i]['close']
        tp = preco_entrada * (1 + ALVO_LUCRO)
        stop = preco_entrada * (1 - ALVO_STOP)
        
        resultado = 0
        for j in range(1, FUTURO_VISAO + 1):
            f = df.iloc[i + j]
            if f['low'] <= stop: # Tocou stop
                resultado = 0
                break
            if f['high'] >= tp: # Tocou TP
                resultado = 1
                break
        targets.append(resultado)
    return targets

def main():
    # 1. Baixar
    df = baixar_historico_profundo()
    
    # 2. Calcular Indicadores
    print("🧮 Calculando indicadores (pode demorar um pouco)...")
    df = Calculadora.adicionar_todos(df)
    
    # Remove NaN (os primeiros 200 candles que ficam vazios por causa da EMA)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 3. Criar Alvo
    df['target'] = criar_alvo(df)
    
    # 4. Salvar
    # Removemos as últimas linhas onde não sabemos o futuro
    df = df.iloc[:-FUTURO_VISAO]
    
    # Seleciona colunas úteis
    colunas_finais = [
        'open', 'high', 'low', 'close', 'volume',
        'RSI_14', 'ADX_14', 'ATRr_14', 'BBU_20_2.0', 'BBL_20_2.0',
        'EMA_9', 'EMA_21', 'EMA_200', 'DIST_EMA21',
        'target'
    ]
    # Filtra só o que existe (para evitar erro)
    cols_existentes = [c for c in colunas_finais if c in df.columns]
    
    df[cols_existentes].to_csv("dataset_grande.csv", index=False)
    
    wins = df[df['target'] == 1].shape[0]
    total = df.shape[0]
    print(f"\n💾 ARQUIVO GERADO: dataset_grande.csv")
    print(f"📊 Total de Dados Prontos: {total}")
    print(f"🏆 Oportunidades de Win encontradas: {wins} ({(wins/total)*100:.2f}%)")

if __name__ == "__main__":
    main()