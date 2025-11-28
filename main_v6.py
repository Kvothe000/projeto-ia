# Binance/main_v7.py - O GESTOR DE 1% DIÁRIO (FUSÃO V6 + GESTÃO)
import time
import pandas_ta as ta
from binance_connector import BinanceConnector
from ai_trader_v6 import TraderIAV6
from manager import GerenciadorEstado
from scanner import ScannerCrypto

class DailyGoalBot:
    def __init__(self):
        # --- CONFIGURAÇÕES DE OURO (1% DIÁRIO) ---
        self.META_DIARIA_PCT = 0.015    # Meta: 1.5% (para garantir 1% líquido)
        self.STOP_DIARIO_PCT = -0.01    # Stop Diário: -1% (Trava de segurança)
        self.VALOR_BANCA = 200          # Sua banca total alocada
        self.ALAVANCAGEM = 5
        
        # Gestão de Trade
        self.VALOR_APOSTA = self.VALOR_BANCA * self.ALAVANCAGEM # Power
        self.GATILHO_TRAILING = 0.005   # Ativa trailing com 0.5% de lucro
        self.CALLBACK_TRAILING = 0.002  # Devolve 0.2% e sai
        
        # Componentes
        self.con = BinanceConnector()
        self.cerebro = TraderIAV6()
        self.gerenciador = GerenciadorEstado()
        self.scanner = ScannerCrypto()
        
        # Estado do Dia
        self.lucro_acumulado = 0.0
        self.trades_hoje = 0
        self.reset_dia()

    def reset_dia(self):
        # Lógica simples: reseta se mudar o dia (pode melhorar depois)
        self.ultimo_dia = time.localtime().tm_mday
        self.lucro_acumulado = 0.0
        self.trades_hoje = 0
        print("☀️ Novo dia iniciado! Metas resetadas.")

    def verificar_metas(self):
        # Verifica virada do dia
        if time.localtime().tm_mday != self.ultimo_dia:
            self.reset_dia()
            
        # Verifica Meta Batida
        pct_hoje = self.lucro_acumulado / self.VALOR_BANCA
        if pct_hoje >= self.META_DIARIA_PCT:
            print(f"🎉 META BATIDA! Lucro: {pct_hoje*100:.2f}%. Bot indo dormir...")
            return False # Para o bot
            
        # Verifica Stop Loss Diário
        if pct_hoje <= self.STOP_DIARIO_PCT:
            print(f"💀 STOP DIÁRIO! Prejuízo: {pct_hoje*100:.2f}%. Bot parando por hoje.")
            return False # Para o bot
            
        print(f"📊 Status Dia: {pct_hoje*100:.2f}% (Meta: {self.META_DIARIA_PCT*100}%)")
        return True # Continua operando

    def filtro_rsi(self, df, sinal):
        """Filtro extra sugerido pelo colega (Evita comprar topo)"""
        df.ta.rsi(length=14, append=True)
        rsi = df.iloc[-1]['RSI_14']
        
        if sinal == "BUY" and rsi > 70:
            return False, f"RSI Esticado ({rsi:.0f})"
        if sinal == "SELL" and rsi < 30:
            return False, f"RSI Esticado ({rsi:.0f})"
            
        return True, "OK"

    def run(self):
        print("🤖 BOT V7 INICIADO: MODO GESTOR DE 1%...")
        
        em_posicao = False
        par_atual = None
        ultimo_scan = 0
        
        # Dados do Trade Aberto
        lado_trade = None
        preco_entrada = 0
        max_preco = 0
        
        while True:
            try:
                if not self.verificar_metas():
                    time.sleep(3600) # Dorme 1h se bateu meta
                    continue

                agora = time.time()

                # --- 1. SCANNER (A cada 10 min ou se livre) ---
                if not em_posicao and (agora - ultimo_scan > 600):
                    print("\n🛰️ Atualizando Radar (10min)...")
                    lista = self.scanner.mostrar_top_oportunidades()
                    if not lista: lista = ['WLDUSDT', '1000PEPEUSDT'] # Fallback
                    par_atual = lista[0] # Foca na #1
                    ultimo_scan = agora
                    print(f"🔭 Alvo: {par_atual}")

                # --- 2. ANÁLISE ---
                time.sleep(1)
                df = self.con.buscar_candles(par_atual, "15m", limit=500)
                if df is None: continue
                preco = df.iloc[-1]['close']

                # IA V6 (Com Contexto)
                sinal, confianca, atr_pct = self.cerebro.analisar_mercado(df)
                
                # Visual
                cor = "\033[93m"
                if sinal == "BUY": cor = "\033[92m"
                if sinal == "SELL": cor = "\033[91m"
                print(f"🧠 {par_atual}: {cor}{sinal} ({confianca*100:.1f}%){'\033[0m'} | ATR: {atr_pct:.2f}%")

                # --- 3. ENTRADA ---
                if not em_posicao and sinal in ["BUY", "SELL"]:
                    if self.gerenciador.pode_enviar_alerta(par_atual, "15m"):
                        
                        # Filtro RSI (Extra)
                        ok_rsi, msg_rsi = self.filtro_rsi(df, sinal)
                        if not ok_rsi:
                            print(f"⚠️ Sinal Ignorado: {msg_rsi}")
                            continue

                        # Se passou tudo, ATACA!
                        print(f"\n🚀 SINAL V7 CONFIRMADO!")
                        
                        # Lógica de Quantidade
                        preco_book = self.con.buscar_melhor_preco_book(par_atual, sinal) or preco
                        qtd = self.con.calcular_qtd_correta(par_atual, self.VALOR_APOSTA, preco_book)
                        
                        if qtd > 0:
                            print(f"💰 Entrada: {preco_book} | Alavancagem: {self.ALAVANCAGEM}x")
                            print("✅ Ordem Simulada Preenchida!")
                            
                            em_posicao = True
                            lado_trade = sinal
                            preco_entrada = preco_book
                            max_preco = preco_book
                            
                            self.gerenciador.registrar_envio(par_atual)
                            self.gerenciador.registrar_trade(par_atual, sinal, preco_book, qtd, self.VALOR_APOSTA, "V7-AGRESSIVO")

                # --- 4. GESTÃO (TRAILING AGRESSIVO) ---
                if em_posicao:
                    # Calcula Lucro Atual
                    if lado_trade == "BUY":
                        lucro_pct = (preco - preco_entrada) / preco_entrada
                        if preco > max_preco: max_preco = preco
                        queda = (max_preco - preco) / max_preco
                    else:
                        lucro_pct = (preco_entrada - preco) / preco_entrada
                        if preco < max_preco: max_preco = preco # Menor é melhor no short
                        queda = (preco - max_preco) / max_preco

                    print(f"   🛡️ Posição: {lucro_pct*100:.2f}% | Max: {max_preco}")

                    # Regras de Saída V7
                    sair = False
                    motivo = ""

                    # 1. Stop Loss de Emergência (Fixo ATR 2x ou -0.6%)
                    if lucro_pct < -0.006: 
                        sair = True; motivo = "STOP LOSS"
                    
                    # 2. Trailing Stop (Garantir Lucro)
                    elif lucro_pct > self.GATILHO_TRAILING: # Já lucrou 0.5%?
                        print(f"      🎯 Trailing Ativado! (Devolver {self.CALLBACK_TRAILING*100}%)")
                        if queda > self.CALLBACK_TRAILING:
                            sair = True; motivo = f"TRAILING PROFIT (+{lucro_pct*100:.2f}%)"

                    if sair:
                        print(f"👋 SAÍDA: {motivo}")
                        lucro_usdt = self.VALOR_APOSTA * lucro_pct
                        self.lucro_acumulado += lucro_usdt
                        self.trades_hoje += 1
                        
                        em_posicao = False
                        par_atual = None # Força rescan
                        print(f"💰 Resultado Trade: ${lucro_usdt:.2f}")
                        time.sleep(2)

                time.sleep(5) # Ciclo rápido

            except KeyboardInterrupt:
                print("\n🛑 Bot parado.")
                break
            except Exception as e:
                print(f"❌ Erro: {e}")
                time.sleep(5)

if __name__ == "__main__":
    bot = DailyGoalBot()
    bot.run()