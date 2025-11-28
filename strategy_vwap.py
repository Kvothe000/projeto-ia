# Binance/strategy_vwap.py
import pandas as pd
import pandas_ta as ta
import time
from binance_connector import BinanceConnector
from manager import GerenciadorEstado

# --- CONFIGURAÇÕES DO SNIPER ---
ALVO_LUCRO = 0.012  # 1.2% (Busca o 1% líquido)
STOP_LOSS = 0.006   # 0.6% (Risco metade do lucro)
ALAVANCAGEM = 5     # 5x (Moderada)

class VwapPredator:
    def __init__(self, par_alvo):
        self.par = par_alvo
        self.connector = BinanceConnector()
        self.gerenciador = GerenciadorEstado()
        print(f"🐅 PREDADOR INICIADO EM: {self.par}")

    def calcular_vwap_diaria(self, df):
        """
        Calcula a VWAP Ancorada (Reinicia todo dia à 00:00 UTC).
        Essencial para Day Trade.
        """
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['pv'] = df['close'] * df['volume']
        
        # Agrupa por dia para reiniciar o cálculo
        # A VWAP é: SomaAcumulada(Preço*Vol) / SomaAcumulada(Vol)
        df['vwap_m'] = df.groupby(df['timestamp'].dt.date)['pv'].cumsum()
        df['vol_m'] = df.groupby(df['timestamp'].dt.date)['volume'].cumsum()
        df['VWAP'] = df['vwap_m'] / df['vol_m']
        
        return df

    def analisar_oportunidade(self):
        print(f"\n🔍 Analisando {self.par} (15m)...")
        
        # 1. Buscar Dados de Preço e Volume
        df = self.connector.buscar_candles(self.par, '15m', mercado="FUTUROS", limit=200)
        if df is None: return
        
        # 2. Buscar Open Interest (O Dinheiro na Mesa)
        oi_df = self.connector.buscar_open_interest(self.par, '15m')
        
        # 3. Calcular Indicadores
        df = self.calcular_vwap_diaria(df)
        
        atual = df.iloc[-1]
        anterior = df.iloc[-2]
        
        preco = atual['close']
        vwap = atual['VWAP']
        cvd = atual['CVD']
        
        # Variação do Open Interest (Dinheiro entrando ou saindo?)
        if oi_df is not None and not oi_df.empty:
            oi_atual = oi_df.iloc[-1]['sumOpenInterest']
            oi_anterior = oi_df.iloc[-5]['sumOpenInterest'] # Comparando com 1 hora atrás
            delta_oi = (oi_atual - oi_anterior) / oi_anterior * 100
        else:
            delta_oi = 0 # Sem dados

        print(f"📊 Preço: {preco:.4f} | VWAP: {vwap:.4f}")
        print(f"💰 CVD (Fluxo): {cvd:.0f} | OI (Interesse): {delta_oi:.2f}%")

        # --- LÓGICA DE COMPRA (LONG) ---
        # 1. Preço ACIMA da VWAP (Tendência de Alta Intraday)
        # 2. CVD Positivo (Compradores agredindo)
        # 3. OI Subindo (Dinheiro novo entrando = movimento forte)
        if preco > vwap and cvd > 0 and delta_oi > 1.0:
            print("✅ SETUP LONG: Preço > VWAP + Fluxo Comprador!")
            
            # Verifica se rompeu a VWAP agora (Gatilho) ou Pullback
            distancia_vwap = (preco - vwap) / vwap * 100
            
            if distancia_vwap < 0.5: # Perto da VWAP (Ponto ideal de entrada)
                self.executar_trade("BUY")
            else:
                print(f"⚠️ Muito esticado ({distancia_vwap:.2f}% da VWAP). Esperar retorno.")

        # --- LÓGICA DE VENDA (SHORT) ---
        elif preco < vwap and cvd < 0 and delta_oi > 0.3:
            print("🔻 SETUP SHORT: Preço < VWAP + Fluxo Vendedor!")
            
            distancia_vwap = (vwap - preco) / preco * 100
            if distancia_vwap < 0.5:
                self.executar_trade("SELL")
            else:
                print(f"⚠️ Muito esticado ({distancia_vwap:.2f}% da VWAP). Esperar retorno.")
        else:
            print("💤 Mercado Indeciso. Aguardando alinhamento de Fluxo.")

    def executar_trade(self, lado):
        if not self.gerenciador.pode_enviar_alerta(self.par, '15m'):
            print("⏳ Cooldown ativo. Trade ignorado.")
            return

        print(f"🚀 EXECUTANDO ORDEM {lado} (Simulação)!")
        print(f"🎯 Alvo: {ALVO_LUCRO*100}% | Stop: {STOP_LOSS*100}%")
        
        # AQUI ENTRARIA O COMANDO REAL:
        # qtd = calcular_tamanho_posicao(...)
        # self.connector.colocar_ordem_futuros(self.par, lado, qtd, ALAVANCAGEM)
        
        self.gerenciador.registrar_envio(self.par)

# --- LOOP PRINCIPAL ---
if __name__ == "__main__":
    # 1. Primeiro rodamos o scanner para decidir a moeda
    try:
        from scanner import ScannerCrypto
        scanner = ScannerCrypto()
        melhor_moeda = scanner.mostrar_top_oportunidades()
        
        if melhor_moeda:
            # 2. Iniciamos o Predador na moeda escolhida
            bot = VwapPredator(melhor_moeda)
            
            while True:
                try:
                    bot.analisar_oportunidade()
                    print("--- Esperando 30s ---")
                    time.sleep(30)
                except KeyboardInterrupt:
                    print("🛑 Parando...")
                    break
                except Exception as e:
                    print(f"❌ Erro no loop: {e}")
                    time.sleep(10)
    except Exception as e:
        print(f"Erro ao iniciar: {e}")