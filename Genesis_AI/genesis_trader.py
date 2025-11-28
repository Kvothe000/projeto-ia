# Genesis_AI/genesis_trader.py - O EXECUTOR (LIVE TRADING) - REFATORADO
import time
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from binance_connector import BinanceConnector
import sys
import os

# Adiciona o diretório pai ao path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Binance')))
from features_engine import FeaturesEngine

# --- CONFIGURAÇÃO ---
MODELO_PATH = "cerebros/genesis_v2_stable"
PAR_ALVO = "WLDUSDT"
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
        self.posicao = 0  # 0=Neutro, 1=Long, -1=Short
        
        # Carrega estatísticas de normalização do dataset de treino
        self._carregar_parametros_normalizacao()

    def _carregar_parametros_normalizacao(self):
        """Carrega médias e desvios padrão do dataset de treino para normalização"""
        try:
            df_ref = pd.read_csv('../Binance/dataset_v11_fusion.csv')
            
            # Seleciona apenas as colunas numéricas que o modelo espera
            colunas_modelo = FeaturesEngine.colunas_finais()
            df_ref = df_ref[colunas_modelo]
            
            self.mean = df_ref.mean()
            self.std = df_ref.std()
            print("📊 Parâmetros de normalização carregados.")
            print(f"📈 {len(colunas_modelo)} features: {colunas_modelo}")
        except Exception as e:
            print(f"⚠️ Aviso: Erro ao carregar dataset de referência: {e}")
            print("🚨 Usando normalização padrão (pode afetar performance)")
            self.mean = 0
            self.std = 1

    def preparar_dados_live(self, df_moeda):
        """Prepara dados para inferência usando o mesmo processamento do treino"""
        try:
            # 1. Baixa BTC (Contexto) - mesmo período
            df_btc = self.con.buscar_candles("BTCUSDT", TIMEFRAME, limit=len(df_moeda))
            if df_btc is None:
                print("❌ Falha ao carregar dados do BTC")
                return None

            # 2. Processa usando o MESMO motor do treino
            df_proc = FeaturesEngine.processar_dados(df_moeda, df_btc)
            
            # 3. Seleciona colunas que o modelo espera
            colunas_alvo = FeaturesEngine.colunas_finais()
            
            # Verifica se todas as colunas necessárias estão presentes
            colunas_faltantes = set(colunas_alvo) - set(df_proc.columns)
            if colunas_faltantes:
                print(f"❌ Colunas faltantes: {colunas_faltantes}")
                return None
            
            X = df_proc[colunas_alvo].iloc[[-1]]  # Pega última linha
            
            # 4. Normalização (Z-Score) usando parâmetros do treino
            X_norm = (X - self.mean) / self.std
            X_norm = X_norm.fillna(0).clip(-5, 5)
            
            return X_norm.values.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Erro no preparo de dados: {e}")
            return None

    def executar_ordem(self, acao):
        """Executa ordem baseada na decisão da IA"""
        if acao == 1 and self.posicao != 1:  # COMPRAR
            print("🚀 ORDEM: COMPRAR!")
            # self.con.colocar_ordem(PAR_ALVO, "BUY", CAPITAL_TRADE)
            self.posicao = 1
            
        elif acao == 2 and self.posicao != -1:  # VENDER
            print("🔻 ORDEM: VENDER!")
            # self.con.colocar_ordem(PAR_ALVO, "SELL", CAPITAL_TRADE)
            self.posicao = -1
            
        elif acao == 3 and self.posicao != 0:  # FECHAR
            print("🛡️ ORDEM: FECHAR POSIÇÃO!")
            # self.con.fechar_posicao(PAR_ALVO)
            self.posicao = 0
            
        else:
            print(f"⚡ MANTER: Posição atual {self.posicao}")

    def run(self):
        """Loop principal de trading"""
        print(f"🔭 Observando {PAR_ALVO} no timeframe {TIMEFRAME}...")
        print("💡 Modo: SIMULAÇÃO (ordens não são executadas)")
        
        contador_ciclos = 0
        
        try:
            while True:
                contador_ciclos += 1
                print(f"\n📊 Ciclo #{contador_ciclos} - {time.strftime('%H:%M:%S')}")
                
                # 1. Baixa Dados da Moeda
                df_moeda = self.con.buscar_candles(PAR_ALVO, TIMEFRAME, limit=100)
                if df_moeda is None or len(df_moeda) < 50:
                    print("⏳ Aguardando dados...")
                    time.sleep(10)
                    continue

                # 2. Prepara Observação
                obs = self.preparar_dados_live(df_moeda)
                if obs is None:
                    time.sleep(10)
                    continue

                # 3. IA Decide
                acao, _states = self.model.predict(obs, deterministic=True)
                acao = int(acao[0]) if isinstance(acao, np.ndarray) else int(acao)
                
                # 4. Log da Decisão
                acoes = {0: "AGUARDAR", 1: "COMPRAR", 2: "VENDER", 3: "FECHAR"}
                print(f"🧠 Gênesis: {acoes.get(acao, f'Ação {acao}')}")
                print(f"💰 Posição atual: {self.posicao}")

                # 5. Executa (Simulação)
                self.executar_ordem(acao)

                # 6. Aguarda próximo ciclo
                print("⏰ Aguardando próximo candle...")
                time.sleep(13)  # Para timeframe 15m

        except KeyboardInterrupt:
            print("\n🛑 Gênesis Parado pelo usuário.")
        except Exception as e:
            print(f"❌ Erro crítico: {e}")
        finally:
            print("🧹 Finalizando...")
            if self.posicao != 0:
                print("⚠️ ATENÇÃO: Posição ainda aberta!")


if __name__ == "__main__":
    bot = GenesisTrader()
    bot.run()