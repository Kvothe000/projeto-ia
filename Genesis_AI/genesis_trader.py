# Genesis_AI/genesis_trader.py - O EXECUTOR (LIVE TRADING)
import time
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from binance_connector import BinanceConnector # Reutilizamos o nosso conector robusto
import sys
import os

# Adiciona o diretório pai ao path para importar módulos da pasta Binance
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Binance')))
from indicators import Calculadora

# --- CONFIGURAÇÃO ---
MODELO_PATH = "cerebros/genesis_v2_stable"
PAR_ALVO = "WLDUSDT" # O Gênesis pode operar qualquer um, mas vamos testar na WLD
TIMEFRAME = "15m"
CAPITAL_TRADE = 200

class GenesisTrader:
    def __init__(self):
        print("🧬 INICIANDO GÊNESIS LIVE TRADER...")
        
        # 1. Carrega Cérebro
        if os.path.exists(MODELO_PATH + ".zip"):
            self.model = PPO.load(MODELO_PATH)
            print("🧠 Cérebro carregado com sucesso!")
        else:
            print("❌ Erro: Modelo não encontrado.")
            exit()
            
        self.con = BinanceConnector()
        self.posicao = 0 # 0=Neutro, 1=Long, -1=Short
        
        # Carrega estatísticas de normalização (Média/Desvio) do dataset de treino
        # Isso é crucial: A IA precisa ver os dados na mesma escala que treinou!
        try:
            df_ref = pd.read_csv('../Binance/dataset_v11_fusion.csv')
            df_ref = df_ref.select_dtypes(include=[np.number])
            self.mean = df_ref.mean()
            self.std = df_ref.std()
            print("📊 Parâmetros de normalização carregados.")
        except:
            print("⚠️ Aviso: Dataset de referência não encontrado. Normalização pode falhar.")
            self.mean = 0
            self.std = 1

    def preparar_dados_live(self, df):
        # Garante que temos as mesmas features do treino
        # Assume que o df já vem com indicadores do conector ou calcula aqui
        # Para simplificar, vamos assumir que o dataset_v11_fusion.csv foi gerado
        # com colunas que sabemos calcular.
        
        # Recalcula indicadores básicos (caso venha cru)
        df = Calculadora.adicionar_todos(df)
        
        # ... (Adicionar lógica de features V11 Fusion aqui se necessário) ...
        # Como o treino usou o dataset V11 Fusion, precisamos recriar EXATAMENTE
        # as mesmas colunas.
        # Simplificação: Vamos assumir que o conector já traz ou calculamos rápido
        # SE AS FEATURES NÃO BATEREM, A IA VAI ERRAR.
        
        # Seleciona apenas numéricos
        df = df.select_dtypes(include=[np.number])
        
        # Normaliza (Z-Score) usando a referência do treino
        df_norm = (df - self.mean) / self.std
        df_norm = df_norm.fillna(0).clip(-5, 5)
        
        # Retorna última linha como observação
        obs = df_norm.iloc[-1].values.astype(np.float32)
        return obs

    def run(self):
        print(f"🔭 Observando {PAR_ALVO}...")
        
        while True:
            try:
                time.sleep(2) # Loop rápido
                
                # 1. Baixa Dados
                df = self.con.buscar_candles(PAR_ALVO, TIMEFRAME, limit=100) # Precisa de histórico p/ indicadores
                if df is None: continue
                
                # 2. Prepara Observação (Normalização)
                # Nota: Precisamos garantir que as colunas do DF sejam IGUAIS ao treino
                # Isso requer que o 'binance_connector' ou uma função auxiliar
                # gere as features 'mom_3', 'vol_ratio', etc.
                # VAMOS PRECISAR DO 'gerar_dataset_v11_fusion.py' LOGIC AQUI.
                # (Vou simplificar assumindo que você vai copiar a função 'criar_features_avancadas' pra cá
                # ou importar. Por enquanto, deixo o esqueleto).
                
                # [AQUI ENTRA A LÓGICA DE FEATURES IGUAL AO TREINO]
                # ...
                
                obs = self.preparar_dados_live(df) # Placeholder
                
                # 3. IA Decide
                action, _ = self.model.predict(obs, deterministic=True)
                
                # 4. Execução
                print(f"🧠 Gênesis diz: Ação {action}")
                
                if action == 1 and self.posicao != 1:
                    print("🚀 COMPRAR!")
                    # self.con.colocar_ordem(...)
                    self.posicao = 1
                    
                elif action == 2 and self.posicao != -1:
                    print("🔻 VENDER!")
                    # self.con.colocar_ordem(...)
                    self.posicao = -1
                    
                elif action == 3 and self.posicao != 0:
                    print("🛡️ FECHAR!")
                    self.posicao = 0
                
                time.sleep(13) # Espera próximo candle (aprox)

            except KeyboardInterrupt:
                print("🛑 Gênesis Parado.")
                break
            except Exception as e:
                print(f"❌ Erro: {e}")
                time.sleep(5)

if __name__ == "__main__":
    bot = GenesisTrader()
    bot.run()