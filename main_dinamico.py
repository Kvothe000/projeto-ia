# Binance/main_dinamico.py
import time
from binance_connector import BinanceConnector
from ai_trader_v7 import TraderIAV5
from manager import GerenciadorEstado
from scanner import ScannerCrypto

# --- CONFIGURAÇÕES ---
TIMEFRAME = "15m"
INTERVALO_RESCAN = 30 * 60  # Re-escaneia o mercado a cada 30 minutos

def main():
    print("🤖 INICIANDO BOT DINÂMICO (IA V5 + SCANNER)...")
    
    # Inicializa os Módulos
    connector = BinanceConnector()
    cerebro = TraderIAV5()
    gerenciador = GerenciadorEstado()
    scanner = ScannerCrypto()
    
    # Variáveis de Controle
    par_atual = None
    ultimo_scan = 0
    
    # Validação Inicial
    try:
        connector.client.ping()
        print("✅ Conectado à Binance Futures")
    except:
        print("❌ Falha na conexão API")
        return

    while True:
        try:
            # --- 1. MÓDULO SCANNER (O Cérebro Estratégico) ---
            # Verifica se está na hora de mudar de moeda
            agora = time.time()
            if agora - ultimo_scan > INTERVALO_RESCAN:
                print("\n🛰️ Atualizando Radar de Oportunidades...")
                nova_rainha = scanner.mostrar_top_oportunidades()
                
                if nova_rainha and nova_rainha != par_atual:
                    print(f"🔄 TROCA DE ALVO: Saindo de {par_atual} -> Entrando em {nova_rainha}")
                    par_atual = nova_rainha
                    # Limpa histórico ou estados se necessário
                elif not nova_rainha:
                    print("⚠️ Scanner não encontrou nada. Mantendo anterior ou aguardando.")
                
                ultimo_scan = agora
                print(f"🔭 Alvo Travado: {par_atual} (Monitorando com IA V5)...")

            if not par_atual:
                time.sleep(5)
                continue

            # --- 2. MÓDULO SNIPER (A Execução Tática) ---
            # Baixa dados da moeda escolhida pelo Scanner
            df = connector.buscar_candles(par_atual, TIMEFRAME, mercado="FUTUROS", limit=500)
            
            if df is not None:
                preco_atual = df.iloc[-1]['close']
                
                # Pergunta à IA
                sinal, confianca = cerebro.analisar_mercado(df)
                
                # Cores para o log
                cor = "\033[93m" # Amarelo
                if sinal == "BUY": cor = "\033[92m" # Verde
                if sinal == "SELL": cor = "\033[91m" # Vermelho
                reset = "\033[0m"
                
                print(f"🧠 IA em {par_atual}: {cor}{sinal} ({confianca*100:.1f}%){reset} | Preço: {preco_atual}")

                # Se a IA der sinal, verifica se podemos atirar
                if sinal in ["BUY", "SELL"]:
                    if gerenciador.pode_enviar_alerta(par_atual, TIMEFRAME):
                        print(f"\n🚀 {cor}SNIPER SHOT NA {par_atual}!{reset}")
                        print(f"⚡ Ordem: {sinal} | Confiança: {confianca*100:.1f}%")
                        
                        # Cálculo Dinâmico de Stop (ATR) seria ideal aqui
                        # Por enquanto, mantemos fixo ou usamos a lógica do strategy_vwap
                        print(f"🎯 Executando trade na volatilidade máxima...")
                        
                        # --- ORDEM REAL VIRIA AQUI ---
                        # connector.colocar_ordem_futuros(...)
                        
                        gerenciador.registrar_envio(par_atual)
                    else:
                        print("⏳ Cooldown ativo.")
            
            time.sleep(10) # Tick do Robô
            
        except KeyboardInterrupt:
            print("\n🛑 Bot parado.")
            break
        except Exception as e:
            print(f"❌ Erro no loop principal: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()