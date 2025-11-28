# Binance/main_v7.py - VERSÃO CORRIGIDA (BUG FIX)
import time
import pandas_ta as ta
from binance_connector import BinanceConnector
from ai_trader_v6 import TraderIAV6
from manager import GerenciadorEstado
from scanner import ScannerCrypto
from technical_analyzer import AnalisadorTecnico

class TradingBotV7:
    def __init__(self):
        # --- CONFIGURAÇÕES FINANCEIRAS (O Segredo do 1%) ---
        self.CAPITAL_TOTAL = 1000.0     # Seu Capital Total
        self.META_DIARIA_PCT = 0.01     # Meta: 1% ao dia
        self.STOP_DIARIO_PCT = -0.005   # Stop: -0.5%
        
        # Gestão de Risco por Trade
        self.VALOR_BASE_TRADE = 200     # Aposta padrão
        self.ALAVANCAGEM = 5
        self.TIMEFRAME_PRINCIPAL = "15m"
        
        # Componentes
        self.con = BinanceConnector()
        self.cerebro = TraderIAV6()
        self.gerenciador = GerenciadorEstado()
        self.scanner = ScannerCrypto()
        self.tecnico = AnalisadorTecnico()
        
        # Controle Diário
        self.lucro_hoje_usdt = 0.0
        self.trades_hoje = 0
        self.ultimo_dia_operado = time.localtime().tm_mday
        
        # Estado Operacional
        self.em_posicao = False
        self.ultimo_scan = 0
        
        # Dados do Trade Aberto
        self.par_operacao = None
        self.lado_trade = None
        self.preco_entrada = 0
        self.preco_stop = 0
        self.preco_alvo = 0
        self.max_preco = 0

    def verificar_meta_diaria(self):
        """Verifica se já batemos a meta ou o stop do dia"""
        hoje = time.localtime().tm_mday
        
        # Reset diário
        if hoje != self.ultimo_dia_operado:
            print(f"\n☀️ NOVO DIA! Resetando métricas (Dia {hoje})...")
            self.lucro_hoje_usdt = 0.0
            self.trades_hoje = 0
            self.ultimo_dia_operado = hoje
            
        meta_usdt = self.CAPITAL_TOTAL * self.META_DIARIA_PCT
        stop_usdt = self.CAPITAL_TOTAL * self.STOP_DIARIO_PCT
        
        # Verifica Meta
        if self.lucro_hoje_usdt >= meta_usdt:
            print(f"🎉 META BATIDA! Lucro: ${self.lucro_hoje_usdt:.2f}. Bot descansando...")
            return False
            
        # Verifica Stop Loss
        if self.lucro_hoje_usdt <= stop_usdt:
            print(f"💀 STOP LOSS DIÁRIO! Prejuízo: ${self.lucro_hoje_usdt:.2f}. Volte amanhã.")
            return False
            
        # Mostra status
        print(f"📊 Status Dia: ${self.lucro_hoje_usdt:.2f} (Meta: ${meta_usdt:.2f}) | Trades: {self.trades_hoje}")
        return True

    def calcular_mao_dinamica(self, confianca, volatilidade_atr):
        """Kelly Criterion Simplificado"""
        fator_confianca = 1.0
        if confianca >= 0.70: fator_confianca = 1.5
        elif confianca <= 0.55: fator_confianca = 0.7
        
        fator_volatilidade = 1.0
        if volatilidade_atr > 5.0: fator_volatilidade = 0.6 
        
        qtd_usdt = self.VALOR_BASE_TRADE * fator_confianca * fator_volatilidade
        return max(50, min(qtd_usdt, 400)) 

    def analise_multitimeframe(self, par, df_btc_cache):
        """Analisa 1h, 15m e 5m"""
        pesos = {"1h": 0.4, "15m": 0.4, "5m": 0.2}
        score_buy = 0
        score_sell = 0
        atr_15m = 0
        dados_dashboard = {}

        for tf, peso in pesos.items():
            df = self.con.buscar_candles(par, tf, limit=500)
            if df is None: continue
            
            # Cache BTC
            if tf == "15m":
                df_btc = df_btc_cache
            else:
                time.sleep(0.2)
                df_btc = self.con.buscar_candles("BTCUSDT", tf, limit=500)

            # 1. IA V6
            sinal, confianca, atr = self.cerebro.analisar_mercado(df, df_btc)
            
            # 2. Técnica
            df = self.tecnico.adicionar_indicadores_completos(df)
            momentum = self.tecnico.calcular_momentum_score(df)
            
            # Pontuação
            fator = confianca * peso * (1 + momentum)
            
            if sinal == "BUY": score_buy += fator
            elif sinal == "SELL": score_sell += fator
            
            if tf == self.TIMEFRAME_PRINCIPAL:
                atr_15m = atr
                df.ta.adx(length=14, append=True)
                adx = df.iloc[-1].get('ADX_14', 0)
                dados_dashboard = {
                    "par": par,
                    "preco": df.iloc[-1]['close'],
                    "adx": round(adx, 1),
                    "sinal": sinal,
                    "confianca": round(confianca * 100, 1),
                    "status_adx": "V7-PRO"
                }

        if score_buy > 0.60 and score_buy > score_sell:
            return "BUY", score_buy, atr_15m, dados_dashboard
        elif score_sell > 0.60 and score_sell > score_buy:
            return "SELL", score_sell, atr_15m, dados_dashboard
            
        return "NEUTRO", 0, 0, dados_dashboard

    def run(self):
        print("🤖 BOT V7 PRO (1% GOAL) LIGADO!")
        
        while True:
            try:
                # 1. Verifica Metas (CORRIGIDO AQUI)
                if not self.verificar_meta_diaria():
                    time.sleep(3600); continue

                # 2. Scanner
                agora = time.time()
                if not self.em_posicao and (agora - self.ultimo_scan > 600):
                    print("\n🛰️ Atualizando Radar...")
                    lista_alvos = self.scanner.mostrar_top_oportunidades()
                    if not lista_alvos: lista_alvos = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'WLDUSDT']
                    self.ultimo_scan = agora
                    self.candidatos = lista_alvos

                # Garante que candidatos existe
                if not hasattr(self, 'candidatos'): self.candidatos = ['BTCUSDT']
                
                # Foca no trade aberto se houver
                lista_analise = [self.par_operacao] if self.em_posicao else self.candidatos
                relatorio_ciclo = []

                # 3. Baixa BTC Mestre
                df_btc_mestre = self.con.buscar_candles("BTCUSDT", self.TIMEFRAME_PRINCIPAL, limit=500)

                # 4. Ciclo de Análise
                for par in lista_analise:
                    if self.em_posicao and par == self.par_operacao:
                        preco_atual = self.con.obter_preco_atual(par)
                        self.gerir_posicao(preco_atual)
                        relatorio_ciclo.append({
                            "par": par, "preco": preco_atual, "adx": 99, 
                            "sinal": self.lado_trade, "confianca": 100, "status_adx": "EM TRADE"
                        })
                        continue

                    # Se livre
                    time.sleep(1)
                    sinal, score, atr, dados_dash = self.analise_multitimeframe(par, df_btc_mestre)
                    
                    if dados_dash:
                        relatorio_ciclo.append(dados_dash)
                        cor = "\033[93m"
                        if sinal == "BUY": cor = "\033[92m"
                        if sinal == "SELL": cor = "\033[91m"
                        print(f"👀 {par:<12} | Score: {score:.2f} | IA: {cor}{sinal}{'\033[0m'} | ATR: {atr:.2f}%")

                    # EXECUÇÃO
                    if sinal in ["BUY", "SELL"] and not self.em_posicao:
                        if self.gerenciador.pode_enviar_alerta(par, "MULTI"):
                            self.executar_entrada(par, sinal, atr, score)
                            break 

                if relatorio_ciclo:
                    self.gerenciador.atualizar_monitor(relatorio_ciclo)

                time.sleep(5)

            except KeyboardInterrupt:
                print("\n🛑 Bot parado.")
                break
            except Exception as e:
                print(f"❌ Erro Loop: {e}")
                time.sleep(10)

    def executar_entrada(self, par, sinal, atr, score):
        print(f"\n🚀 SINAL V7 CONFIRMADO EM {par}!")
        
        preco = self.con.buscar_melhor_preco_book(par, sinal) or self.con.obter_preco_atual(par)
        valor_trade = self.calcular_mao_dinamica(score/2, atr) 
        qtd = self.con.calcular_qtd_correta(par, valor_trade, preco)
        
        if qtd > 0:
            dist_stop = (atr/100) * preco * 2.0
            dist_alvo = (atr/100) * preco * 3.0
            
            if sinal == "BUY":
                stop = preco - dist_stop
                alvo = preco + dist_alvo
            else:
                stop = preco + dist_stop
                alvo = preco - dist_alvo
                
            print(f"💰 Aposta: ${valor_trade:.0f} | 🛑 Stop: {stop:.4f} | 🎯 Alvo: {alvo:.4f}")
            
            self.em_posicao = True
            self.par_operacao = par
            self.lado_trade = sinal
            self.preco_entrada = preco
            self.preco_stop = stop
            self.preco_alvo = alvo
            self.max_preco = preco
            
            self.gerenciador.registrar_envio(par)
            self.gerenciador.registrar_trade(par, sinal, preco, qtd, valor_trade, "V7-PRO")

    def gerir_posicao(self, preco_atual):
        lucro_pct = (preco_atual - self.preco_entrada) / self.preco_entrada
        if self.lado_trade == "SELL": lucro_pct = -lucro_pct
        
        print(f"   🛡️ {self.par_operacao}: {lucro_pct*100:+.2f}% (Stop: {self.preco_stop})")
        
        saiu = False
        resultado = ""
        
        if self.lado_trade == "BUY":
            if preco_atual <= self.preco_stop: saiu=True; resultado="STOP LOSS"
            elif preco_atual >= self.preco_alvo: saiu=True; resultado="TAKE PROFIT"
        else:
            if preco_atual >= self.preco_stop: saiu=True; resultado="STOP LOSS"
            elif preco_atual <= self.preco_alvo: saiu=True; resultado="TAKE PROFIT"
            
        if saiu:
            lucro_usdt = self.calcular_mao_dinamica(0.6, 1.0) * lucro_pct * self.ALAVANCAGEM 
            self.lucro_hoje_usdt += lucro_usdt
            self.trades_hoje += 1
            
            print(f"👋 SAÍDA: {resultado} | Resultado Liq: ${lucro_usdt:.2f}")
            
            self.em_posicao = False
            self.par_operacao = None
            time.sleep(2)

if __name__ == "__main__":
    bot = TradingBotV7()
    bot.run()