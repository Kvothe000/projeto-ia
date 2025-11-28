# Binance/main_v5.py
import time
from binance_connector import BinanceConnector
from ai_trader_v5 import TraderIAV5
from manager import GerenciadorEstado

# --- CONFIGURAÇÃO ---
PAR_OPERACAO = "WLDUSDT"  # Focamos na moeda que treinamos!
TIMEFRAME = "15m"

def main():
    print(f"🤖 INICIANDO BOT SNIPER V5 - ATIVO: {PAR_OPERACAO}")
    
    connector = BinanceConnector()
    cerebro = TraderIAV5()
    gerenciador = GerenciadorEstado()
    
    # Valida conexão
    try:
        connector.client.ping()
        print("✅ Conectado à Binance Futures")
    except:
        print("❌ Falha na conexão com Binance")
        return

    print(f"🔭 Monitorando mercado (Alvos: Long > {cerebro.limiar_long*100}% | Short > {cerebro.limiar_short*100}%)...")

    while True:
        try:
            # 1. Baixar Candles Recentes (Futuros)
            # Precisamos de pelo menos 200-300 para calcular a VWAP e Médias com precisão
            df = connector.buscar_candles(PAR_OPERACAO, TIMEFRAME, mercado="FUTUROS", limit=500)
            
            if df is not None:
                preco_atual = df.iloc[-1]['close']
                
                # 2. IA Analisa
                sinal, confianca = cerebro.analisar_mercado(df)
                
                # Formatação bonita para o log
                cor = "\033[93m" # Amarelo (Neutro)
                if sinal == "BUY": cor = "\033[92m" # Verde
                if sinal == "SELL": cor = "\033[91m" # Vermelho
                reset = "\033[0m"
                
                print(f"🧠 IA em {PAR_OPERACAO}: {cor}{sinal} ({confianca*100:.1f}%){reset} | Preço: {preco_atual}")

                # 3. Execução
                if sinal in ["BUY", "SELL"]:
                    # Verifica Cooldown (para não fazer spam de ordens)
                    if gerenciador.pode_enviar_alerta(PAR_OPERACAO, TIMEFRAME):
                        print(f"\n🚀 {cor}SNIPER SHOT CONFIRMADO!{reset}")
                        print(f"⚡ Ordem: {sinal} | Confiança: {confianca*100:.1f}%")
                        print(f"🎯 Estratégia: Scalping (Alvo 0.6% / Stop 0.3%)")
                        
                        # --- AQUI ENTRARIA O CÓDIGO DE ORDEM REAL ---
                        # connector.colocar_ordem_futuros(PAR_OPERACAO, sinal, quantidade, alavancagem=5)
                        
                        gerenciador.registrar_envio(PAR_OPERACAO)
                        print("⏳ Entrando em cooldown...")
                    else:
                        print("⏳ Sinal ignorado (Cooldown ativo)")
            
            # Espera 10 segundos para o próximo tick
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot parado.")
            break
        except Exception as e:
            print(f"❌ Erro no loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()