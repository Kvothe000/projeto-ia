# Binance/main_v6.py - O PREDAOR COMPLETO (IA V6 + ATR + DASHBOARD)
import time
import pandas_ta as ta
from binance_connector import BinanceConnector
from ai_trader_v6 import TraderIAV6
from manager import GerenciadorEstado
from scanner import ScannerCrypto

# --- CONFIGURAÇÕES DE RISCO ---
VALOR_APOSTA_USDT = 200   # Valor financeiro por trade
ALAVANCAGEM = 5           
ATR_STOP_MULT = 2.0       # Stop Loss = 2x ATR (Dinâmico)
ATR_TAKE_MULT = 3.0       # Alvo = 3x ATR (Dinâmico)
TIMEFRAME = "15m"
INTERVALO_RESCAN = 30 * 60 # 30 Minutos

def main():
    print("🤖 INICIANDO BOT V6 (INTEGRADO AO DASHBOARD)...")
    
    con = BinanceConnector()
    cerebro = TraderIAV6()
    gerenciador = GerenciadorEstado()
    scanner = ScannerCrypto()
    
    # Estado
    lista_alvos = []
    ultimo_scan = 0
    em_posicao = False
    par_em_operacao = None
    
    # Dados do Trade Aberto
    lado_trade = None
    preco_entrada = 0
    preco_stop_inicial = 0
    preco_alvo_inicial = 0
    
    # Validação
    try:
        con.client.ping()
        print("✅ Conexão Binance OK")
    except:
        print("❌ Falha na Conexão")
        return

    while True:
        try:
            agora = time.time()
            relatorio_ciclo = [] # Para o Dashboard

            # --- 1. SCANNER (Se livre) ---
            if not em_posicao and (agora - ultimo_scan > INTERVALO_RESCAN):
                print("\n🛰️ Atualizando Radar V6...")
                lista_alvos = scanner.mostrar_top_oportunidades()
                if not lista_alvos: lista_alvos = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'WLDUSDT', '1000PEPEUSDT']
                ultimo_scan = agora

            # Decide quem analisar (Lista ou Foco no Trade)
            targets = [par_em_operacao] if em_posicao else lista_alvos

            # --- 2. LOOP DE ANÁLISE ---
            for par_atual in targets:
                time.sleep(1) # Respeita API

                # Baixa dados
                df = con.buscar_candles(par_atual, TIMEFRAME, limit=500)
                if df is None: continue
                preco_atual = df.iloc[-1]['close']

                # Calcula ADX apenas para mostrar no Dashboard (Visual)
                df.ta.adx(length=14, append=True)
                adx_atual = df.iloc[-1]['ADX_14']

                # --- INTELIGÊNCIA V6 ---
                sinal, confianca, atr_pct = cerebro.analisar_mercado(df)

                # --- CONEXÃO COM DASHBOARD ---
                status_moeda = {
                    "par": par_atual,
                    "preco": preco_atual,
                    "adx": round(adx_atual, 1),
                    "sinal": sinal,
                    "confianca": round(confianca * 100, 1),
                    "status_adx": "V6" # Marca d'água da versão
                }
                relatorio_ciclo.append(status_moeda)
                
                # Log Terminal
                cor = "\033[93m"
                if sinal == "BUY": cor = "\033[92m"
                if sinal == "SELL": cor = "\033[91m"
                reset = "\033[0m"
                print(f"👀 {par_atual:<12} | IA: {cor}{sinal} ({confianca*100:.1f}%){reset} | ATR: {atr_pct:.2f}%")

                # Registra histórico visual se for relevante
                if confianca > 0.40:
                    gerenciador.registrar_analise(par_atual, preco_atual, round(adx_atual,1), sinal, round(confianca*100, 1))

                # --- 3. EXECUÇÃO (ENTRADA) ---
                if not em_posicao and sinal in ["BUY", "SELL"]:
                    if gerenciador.pode_enviar_alerta(par_atual, TIMEFRAME):
                        print(f"\n🚀 SINAL V6 CONFIRMADO EM {par_atual}!")
                        
                        # 3.1 Busca Preço Maker
                        preco_book = con.buscar_melhor_preco_book(par_atual, sinal)
                        if preco_book == 0: preco_book = preco_atual
                        
                        # 3.2 Calcula Quantidade (Gestão de Banca)
                        qtd = con.calcular_qtd_correta(par_atual, VALOR_APOSTA_USDT, preco_book)
                        
                        if qtd > 0:
                            # 3.3 Calcula Risco ATR
                            distancia_stop = (atr_pct / 100) * preco_book * ATR_STOP_MULT
                            distancia_alvo = (atr_pct / 100) * preco_book * ATR_TAKE_MULT
                            
                            if sinal == "BUY":
                                stop_loss = preco_book - distancia_stop
                                take_profit = preco_book + distancia_alvo
                                lado_saida = "SELL"
                            else: # SHORT
                                stop_loss = preco_book + distancia_stop
                                take_profit = preco_book - distancia_alvo
                                lado_saida = "BUY"

                            # Arredonda preços
                            _, tick = con.obter_precisao_moeda(par_atual)
                            if tick:
                                stop_loss = round(stop_loss / tick) * tick
                                take_profit = round(take_profit / tick) * tick

                            print(f"💰 Entrada: {preco_book} | 🛑 Stop: {stop_loss} | 🎯 Alvo: {take_profit}")
                            
                            # --- EXECUÇÃO SIMULADA (Segurança Primeiro) ---
                            print("✅ Ordem Simulada Preenchida!")
                            em_posicao = True
                            par_em_operacao = par_atual
                            lado_trade = sinal
                            preco_entrada = preco_book
                            preco_stop_inicial = stop_loss
                            preco_alvo_inicial = take_profit
                            
                            # Registra no Dashboard
                            gerenciador.registrar_envio(par_atual)
                            gerenciador.registrar_trade(
                                par_atual, sinal, preco_book, qtd, VALOR_APOSTA_USDT, f"V6-ATR ({ATR_STOP_MULT}x)"
                            )
                            break # Sai do loop para focar no trade

            # Atualiza Dashboard ao vivo
            if relatorio_ciclo:
                gerenciador.atualizar_monitor(relatorio_ciclo)

            # --- 4. GESTÃO (SAÍDA) ---
            if em_posicao:
                print(f"   🛡️ {par_em_operacao}: Posição Aberta | Stop: {preco_stop_inicial} | Alvo: {preco_alvo_inicial}")
                
                # Verifica Saída (Simulação)
                # Na real, a Binance faria isso sozinha com a ordem OCO
                saiu = False
                resultado = ""
                
                if lado_trade == "BUY":
                    if preco_atual <= preco_stop_inicial: saiu=True; resultado="STOP LOSS"
                    elif preco_atual >= preco_alvo_inicial: saiu=True; resultado="TAKE PROFIT"
                else: # SELL
                    if preco_atual >= preco_stop_inicial: saiu=True; resultado="STOP LOSS"
                    elif preco_atual <= preco_alvo_inicial: saiu=True; resultado="TAKE PROFIT"
                
                if saiu:
                    print(f"👋 SAÍDA: {resultado}")
                    em_posicao = False
                    par_em_operacao = None
                    time.sleep(2)

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n🛑 Bot parado.")
            break
        except Exception as e:
            print(f"❌ Erro Loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()