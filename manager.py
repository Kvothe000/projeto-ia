# Binance/manager.py (BANQUEIRO SINCRONIZADO)
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import time
import random

# Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_ESTADO = os.path.join(BASE_DIR, "bot_state.json")
ARQUIVO_WALLET = os.path.join(BASE_DIR, "bot_wallet.json")
ARQUIVO_HISTORICO = os.path.join(BASE_DIR, "trades_history.csv")
ARQUIVO_MONITOR = os.path.join(BASE_DIR, "monitor_live.json")

class GerenciadorEstado:
    def __init__(self, saldo_inicial=None):
        """
        saldo_inicial: Se fornecido (via API Binance), atualiza o cofre.
        """
        self.dados = self._carregar_json(ARQUIVO_ESTADO)
        self._garantir_carteira(saldo_inicial)
        self._inicializar_historico()
        
        # Sincroniza saldo inicial se fornecido (Conexão Real)
        if saldo_inicial is not None:
            self.sincronizar_saldo_real(saldo_inicial)

    def _carregar_json(self, arquivo):
        for _ in range(10):
            if os.path.exists(arquivo):
                try:
                    with open(arquivo, 'r') as f: return json.load(f)
                except: time.sleep(0.1)
            else: return {}
        return {}

    def _salvar_json(self, arquivo, dados):
        for _ in range(10):
            try:
                with open(arquivo, 'w') as f: json.dump(dados, f, indent=4)
                return
            except: time.sleep(random.uniform(0.05, 0.2))

    def _garantir_carteira(self, saldo_inicial):
        if not os.path.exists(ARQUIVO_WALLET):
            val = saldo_inicial if saldo_inicial else 0.0
            # data_referencia serve para saber quando resetar o PnL diário
            carteira = {
                "saldo": val, 
                "saldo_inicial_dia": val, 
                "em_uso": 0.0,
                "data_referencia": datetime.now().strftime("%Y-%m-%d")
            }
            self._salvar_json(ARQUIVO_WALLET, carteira)

    def _inicializar_historico(self):
        if not os.path.exists(ARQUIVO_HISTORICO):
            pd.DataFrame(columns=[
                "data", "par", "lado", "preco", "qtd", "valor_usdt", "pnl_usd", "pnl_pct", "tipo", "saldo_total"
            ]).to_csv(ARQUIVO_HISTORICO, index=False)

    # --- GESTÃO FINANCEIRA ---
    
    def sincronizar_saldo_real(self, saldo_real_binance):
        """Atualiza o cofre com o que realmente tem na Binance"""
        carteira = self._carregar_json(ARQUIVO_WALLET)
        
        # Verifica virada de dia para resetar meta diária
        hoje = datetime.now().strftime("%Y-%m-%d")
        if carteira.get("data_referencia") != hoje:
            print(f"📅 NOVO DIA DETECTADO! Resetando Saldo Inicial de Referência: ${saldo_real_binance:.2f}")
            carteira["saldo_inicial_dia"] = saldo_real_binance
            carteira["data_referencia"] = hoje
        
        # Se não tivermos trades abertos (em_uso == 0), o saldo real é o saldo total
        if carteira.get("em_uso", 0) == 0:
            carteira["saldo"] = saldo_real_binance
        
        # Se for a primeira vez
        if carteira.get("saldo_inicial_dia") == 0:
             carteira["saldo_inicial_dia"] = saldo_real_binance

        self._salvar_json(ARQUIVO_WALLET, carteira)

    def obter_saldo_disponivel(self):
        c = self._carregar_json(ARQUIVO_WALLET)
        return c.get("saldo", 0.0)

    def reservar_capital(self):
        """Pega TUDO (All-In)"""
        carteira = self._carregar_json(ARQUIVO_WALLET)
        disponivel = carteira.get("saldo", 0.0)
        
        if disponivel < 5: return 0.0 # Mínimo da Binance
        
        valor_reserva = disponivel
        carteira["saldo"] = 0.0
        carteira["em_uso"] = valor_reserva
        self._salvar_json(ARQUIVO_WALLET, carteira)
        return valor_reserva

    def devolver_capital(self, valor_retornado):
        """Devolve ao cofre"""
        carteira = self._carregar_json(ARQUIVO_WALLET)
        carteira["saldo"] = valor_retornado
        carteira["em_uso"] = 0.0
        self._salvar_json(ARQUIVO_WALLET, carteira)
        return valor_retornado

    # --- LOGS E DASHBOARD ---
    def registrar_trade(self, par, lado, preco, qtd, valor_usdt, tipo, pnl_usd=0, pnl_pct=0):
        c = self._carregar_json(ARQUIVO_WALLET)
        total = c.get("saldo", 0) + c.get("em_uso", 0)
        
        novo = {
            "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "par": par, "lado": lado, "preco": preco, "qtd": qtd, 
            "valor_usdt": valor_usdt, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, 
            "tipo": tipo, "saldo_total": total
        }
        try:
            df = pd.read_csv(ARQUIVO_HISTORICO)
            df = pd.concat([df, pd.DataFrame([novo])], ignore_index=True)
            df.to_csv(ARQUIVO_HISTORICO, index=False)
        except: pass

    def atualizar_monitor(self, dados_par):
        try:
            monitor = self._carregar_json(ARQUIVO_MONITOR)
            if "moedas" not in monitor: monitor["moedas"] = []
            
            novas = [m for m in monitor["moedas"] if m["par"] != dados_par[0]["par"]]
            novas.extend(dados_par)
            
            c = self._carregar_json(ARQUIVO_WALLET)
            total = c.get("saldo", 0) + c.get("em_uso", 0)
            
            # Adiciona dados de performance diária para o Dashboard
            saldo_ini = c.get("saldo_inicial_dia", total)
            lucro_dia = total - saldo_ini
            pct_dia = (lucro_dia / saldo_ini * 100) if saldo_ini > 0 else 0

            monitor["moedas"] = novas
            monitor["ultima_atualizacao"] = datetime.now().strftime("%H:%M:%S")
            monitor["saldo_total"] = total
            monitor["lucro_dia_usd"] = lucro_dia
            monitor["lucro_dia_pct"] = pct_dia
            
            self._salvar_json(ARQUIVO_MONITOR, monitor)
        except: pass
        
    def pode_enviar_alerta(self, par, timeframe):
        if par in self.dados:
            try:
                ultimo = datetime.fromisoformat(self.dados[par])
                if datetime.now() - ultimo < timedelta(minutes=2): return False
            except: pass
        return True

    def registrar_envio(self, par):
        self.dados[par] = datetime.now().isoformat()
        self._salvar_json(ARQUIVO_ESTADO, self.dados)