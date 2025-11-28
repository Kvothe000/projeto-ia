# Binance/main_final.py - VERSÃO REFATORADA COM LÓGICA HÍBRIDA
import time
import pandas as pd
import pandas_ta as ta
from binance_connector import BinanceConnector
from ai_trader_v7 import TraderIAV5
from manager import GerenciadorEstado
from scanner import ScannerCrypto

class TradingBot:
    def __init__(self):
        # --- CONFIGURAÇÕES ---
        self.TIMEFRAME = "15m"
        self.INTERVALO_RESCAN = 30 * 60
        self.ADX_MINIMO = 20
        self.ALAVANCAGEM = 5
        self.VALOR_APOSTA_USDT = 200
        self.GATILHO_LUCRO = 0.006
        self.CALLBACK_STOP = 0.003
        self.CONFIANCA_ALTA = 0.55
        self.CONFIANCA_MEDIA = 0.40
        self.ADX_EXPLOSIVO = 30
        
        # --- COMPONENTES ---
        self.con = BinanceConnector()
        self.cerebro = TraderIAV5()
        self.gerenciador = GerenciadorEstado()
        self.scanner = ScannerCrypto()
        
        # --- ESTADO DO BOT ---
        self.lista_alvos = []
        self.ultimo_scan = 0
        self.em_posicao = False
        self.par_em_operacao = None
        self.lado_trade = None
        self.preco_entrada = 0
        self.max_preco_atingido = 0
        self.qtd = 0

    def calcular_vwap(self, df):
        """Calcula VWAP para análise híbrida"""
        try:
            vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            return vwap.iloc[-1]
        except:
            return df['close'].iloc[-1]

    def calcular_distancia_vwap(self, preco_atual, vwap):
        """Calcula distância percentual do VWAP"""
        if vwap == 0:
            return 0
        return ((preco_atual - vwap) / vwap) * 100

    def analisar_mercado_hibrido(self, df, preco_atual, adx_atual, sinal_ia, confianca_ia):
        """
        Aplica lógica híbrida: IA + Análise Técnica
        Retorna: (decisao_final, motivo)
        """
        # 1. IA Pura (Alta Confiança)
        if confianca_ia >= self.CONFIANCA_ALTA:
            return sinal_ia, f"IA Confiável ({confianca_ia*100:.1f}%)"

        # 2. Híbrido (IA Média + Tendência Explosiva)
        if confianca_ia >= self.CONFIANCA_MEDIA and adx_atual > self.ADX_EXPLOSIVO:
            vwap = self.calcular_vwap(df)
            dist_vwap = self.calcular_distancia_vwap(preco_atual, vwap)
            
            # Análise para LONG
            if sinal_ia in ["BUY", "NEUTRO"] and 0 < dist_vwap < 1.0:
                return "BUY", f"Híbrido (ADX {adx_atual:.0f}, VWAP +{dist_vwap:.2f}%)"
            
            # Análise para SHORT  
            elif sinal_ia in ["SELL", "NEUTRO"] and -1.0 < dist_vwap < 0:
                return "SELL", f"Híbrido (ADX {adx_atual:.0f}, VWAP {dist_vwap:.2f}%)"

        return None, "Sem oportunidade clara"

    def executar_scan_mercado(self):
        """Executa scan por novas oportunidades"""
        print("\n🛰️ SATÉLITE: Buscando novas oportunidades...")
        self.lista_alvos = self.scanner.mostrar_top_oportunidades()
        
        if self.lista_alvos:
            print(f"📋 ALVOS CARREGADOS: {self.lista_alvos}")
        else:
            print("⚠️ Scanner não encontrou oportunidades")
        
        self.ultimo_scan = time.time()

    def processar_par(self, par_atual):
        """Processa análise e trading para um par específico"""
        time.sleep(1)  # Rate limit
        
        # Obter dados de mercado
        df = self.con.buscar_candles(par_atual, self.TIMEFRAME, mercado="FUTUROS", limit=500)
        if df is None or df.empty:
            return None

        preco_atual = df.iloc[-1]['close']
        
        # Calcular indicadores
        df.ta.adx(length=14, append=True)
        adx_atual = df.iloc[-1]['ADX_14']
        
        # Consultar IA
        sinal_ia, confianca_ia = self.cerebro.analisar_mercado(df)
        
        # Aplicar lógica híbrida
        decisao_final, motivo = self.analisar_mercado_hibrido(df, preco_atual, adx_atual, sinal_ia, confianca_ia)

        # Registrar análise relevante
        if confianca_ia > 0.40 or adx_atual > 20:
            self.gerenciador.registrar_analise(
                par_atual, preco_atual, round(adx_atual, 1), 
                decisao_final or sinal_ia, round(confianca_ia * 100, 1)
            )

        return {
            "par": par_atual,
            "preco": preco_atual,
            "adx": round(adx_atual, 1),
            "sinal_ia": sinal_ia,
            "confianca_ia": round(confianca_ia * 100, 1),
            "decisao_final": decisao_final,
            "motivo": motivo,
            "status_adx": "OK" if adx_atual >= self.ADX_MINIMO else "BAIXO",
            "timestamp": time.time()
        }

    def log_analise_terminal(self, analise):
        """Exibe análise formatada no terminal"""
        par = analise["par"]
        adx = analise["adx"]
        sinal = analise["decisao_final"] or analise["sinal_ia"]
        confianca = analise["confianca_ia"]
        preco = analise["preco"]
        motivo = analise["motivo"]
        status_adx = "✅" if analise["status_adx"] == "OK" else "❌"

        # Cores para terminal
        cor = "\033[93m"  # Neutro (amarelo)
        if sinal == "BUY": 
            cor = "\033[92m"  # Verde
        elif sinal == "SELL": 
            cor = "\033[91m"  # Vermelho
        
        reset = "\033[0m"
        
        # Log principal
        base_log = f"👀 {par:<12} | ADX: {adx} {status_adx} | "
        
        if analise["decisao_final"]:
            base_log += f"DECISÃO: {cor}{sinal} ({confianca:.1f}%){reset} | "
            base_log += f"💰 {preco:.4f} | 📝 {motivo}"
        else:
            base_log += f"IA: {sinal} ({confianca:.1f}%) | "
            base_log += f"💰 {preco:.4f}"

        print(base_log)

    def executar_entrada(self, analise):
        """Executa ordem de entrada baseada na análise"""
        par_atual = analise["par"]
        decisao = analise["decisao_final"]
        motivo = analise["motivo"]
        
        if not self.gerenciador.pode_enviar_alerta(par_atual, self.TIMEFRAME):
            return False

        print(f"\n🚀 OPORTUNIDADE EM {par_atual}! [{motivo}]")
        
        # Calcular preço e quantidade
        preco_alvo = self.con.buscar_melhor_preco_book(par_atual, decisao)
        if preco_alvo == 0:
            preco_alvo = analise["preco"]
        
        qtd = self.con.calcular_qtd_correta(par_atual, self.VALOR_APOSTA_USDT, preco_alvo)
        
        if qtd <= 0:
            return False

        print(f"💰 Apostando ${self.VALOR_APOSTA_USDT} -> {qtd:.6f} {par_atual}")
        print(f"🎯 Enviando Ordem MAKER a {preco_alvo:.4f}...")
        
        # SIMULAÇÃO - Ordem de entrada
        print("✅ Ordem Preenchida (Simulação)!")
        
        # Atualizar estado
        self.em_posicao = True
        self.par_em_operacao = par_atual
        self.lado_trade = decisao
        self.preco_entrada = preco_alvo
        self.max_preco_atingido = preco_alvo
        self.qtd = qtd
        
        # Registrar operação
        self.gerenciador.registrar_envio(par_atual)
        self.gerenciador.registrar_trade(
            par_atual, decisao, preco_alvo, qtd, 
            self.VALOR_APOSTA_USDT, "MAKER"
        )
        
        return True

    def gerenciar_posicao_aberta(self, analise):
        """Gerencia posição em aberto com trailing stop"""
        par_atual = analise["par"]
        preco_atual = analise["preco"]
        
        print(f"   🛡️ GERINDO POSIÇÃO EM {par_atual}...")
        
        # Cálculos de performance
        if self.lado_trade == "BUY":
            if preco_atual > self.max_preco_atingido:
                self.max_preco_atingido = preco_atual
            lucro_atual_pct = (preco_atual - self.preco_entrada) / self.preco_entrada
            queda_do_topo = (self.max_preco_atingido - preco_atual) / self.max_preco_atingido
        else:  # SELL
            if preco_atual < self.max_preco_atingido:
                self.max_preco_atingido = preco_atual
            lucro_atual_pct = (self.preco_entrada - preco_atual) / self.preco_entrada
            queda_do_topo = (preco_atual - self.max_preco_atingido) / self.max_preco_atingido

        print(f"      Lucro Atual: {lucro_atual_pct*100:.2f}% | Topo: {self.max_preco_atingido:.4f}")

        # Atualizar dashboard
        info_trade = {
            "par": self.par_em_operacao,
            "lado": self.lado_trade,
            "entrada": self.preco_entrada,
            "atual": preco_atual,
            "max_atingido": self.max_preco_atingido,
            "lucro_pct": lucro_atual_pct * 100
        }
        self.gerenciador.atualizar_trade_aberto(info_trade)

        # Verificar condições de saída
        return self.verificar_saida_trade(preco_atual, lucro_atual_pct, queda_do_topo)

    def verificar_saida_trade(self, preco_atual, lucro_pct, queda_topo):
        """Verifica condições para saída do trade"""
        # Stop Loss de Emergência
        if lucro_pct < -0.005:
            return True, "STOP LOSS (-0.5%)"

        # Trailing Stop (Lucro)
        if lucro_pct > self.GATILHO_LUCRO and queda_topo > self.CALLBACK_STOP:
            return True, f"TRAILING STOP (+{lucro_pct*100:.2f}%)"

        return False, ""

    def executar_saida(self, preco_saida, motivo):
        """Executa saída do trade"""
        print(f"👋 ENCERRANDO TRADE: {motivo}")
        
        # SIMULAÇÃO - Ordem de saída
        lado_saida = "SELL" if self.lado_trade == "BUY" else "BUY"
        print(f"✅ Ordem de Saída {lado_saida} Executada (Simulação)!")
        
        # Registrar saída
        lucro_final_pct = (
            (preco_saida - self.preco_entrada) / self.preco_entrada * 100 
            if self.lado_trade == "BUY" else
            (self.preco_entrada - preco_saida) / self.preco_entrada * 100
        )
        
        self.gerenciador.registrar_trade_saida(
            self.par_em_operacao, 
            preco_saida, 
            lucro_final_pct,
            motivo
        )
        
        # Resetar estado
        self.em_posicao = False
        self.par_em_operacao = None
        self.lado_trade = None
        self.qtd = 0
        self.gerenciador.limpar_trade_aberto()
        
        print("🔄 Trade Fechado. Voltando ao Scanner...")
        time.sleep(2)

    def executar_ciclo_analise(self):
        """Executa um ciclo completo de análise"""
        relatorio_ciclo = []
        agora = time.time()

        # Scan periódico do mercado
        if not self.em_posicao and (agora - self.ultimo_scan > self.INTERVALO_RESCAN):
            self.executar_scan_mercado()

        if not self.lista_alvos and not self.em_posicao:
            time.sleep(5)
            return

        # Definir pares para análise
        pares_analise = [self.par_em_operacao] if self.em_posicao else self.lista_alvos

        # Processar cada par
        for par_atual in pares_analise:
            analise = self.processar_par(par_atual)
            if not analise:
                continue

            relatorio_ciclo.append(analise)
            self.log_analise_terminal(analise)

            # Atualizar dashboard em tempo real
            self.gerenciador.atualizar_monitor(relatorio_ciclo)

            # Lógica de entrada
            if (not self.em_posicao and 
                analise["decisao_final"] in ["BUY", "SELL"] and 
                analise["status_adx"] == "OK"):
                
                if self.executar_entrada(analise):
                    break  # Para após entrar em uma posição

            # Lógica de gestão de posição
            elif self.em_posicao and par_atual == self.par_em_operacao:
                deve_sair, motivo_saida = self.gerenciar_posicao_aberta(analise)
                if deve_sair:
                    self.executar_saida(analise["preco"], motivo_saida)
                    break

    def run(self):
        """Loop principal do bot"""
        print("🤖 INICIANDO BOT FINAL (VERSÃO REFATORADA - LÓGICA HÍBRIDA)...")
        
        try:
            while True:
                self.executar_ciclo_analise()
                
        except KeyboardInterrupt:
            print("\n🛑 Bot parado pelo usuário.")
        except Exception as e:
            print(f"❌ Erro Crítico: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

def main():
    bot = TradingBot()
    bot.run()

if __name__ == "__main__":
    main()