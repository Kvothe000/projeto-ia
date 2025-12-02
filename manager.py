# Binance/manager.py (VERSÃO BANQUEIRO BLINDADO)
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
    def __init__(self, saldo_inicial=200.0):
        self.dados = self._carregar_json(ARQUIVO_ESTADO)
        self._garantir_carteira(saldo_inicial)
        self._inicializar_historico()

    def _carregar_json(self, arquivo):
        # Tenta ler com retry para evitar conflito de leitura/escrita
        for _ in range(10):
            if os.path.exists(arquivo):
                try:
                    with open(arquivo, 'r') as f: return json.load(f)
                except: time.sleep(0.1)
            else: return {}
        return {}

    def _salvar_json(self, arquivo, dados):
        # Tenta salvar com retry
        for _ in range(10):
            try:
                with open(arquivo, 'w') as f: json.dump(dados, f, indent=4)
                return
            except: time.sleep(random.uniform(0.05, 0.2))

    def _garantir_carteira(self, saldo_inicial):
        if not os.path.exists(ARQUIVO_WALLET):
            # Se não existe, cria. Se existe, RESPEITA o saldo atual.
            carteira = {"saldo": saldo_inicial, "saldo_inicial": saldo_inicial, "em_uso": 0.0}
            self._salvar_json(ARQUIVO_WALLET, carteira)

    def _inicializar_historico(self):
        if not os.path.exists(ARQUIVO_HISTORICO):
            pd.DataFrame(columns=[
                "data", "par", "lado", "preco", "qtd", "valor_usdt", "pnl_usd", "pnl_pct", "tipo", "saldo_total"
            ]).to_csv(ARQUIVO_HISTORICO, index=False)

    # --- GESTÃO FINANCEIRA (O COFRE ÚNICO) ---
    def obter_saldo_disponivel(self):
        """Retorna apenas o dinheiro que NÃO está em uso"""
        c = self._carregar_json(ARQUIVO_WALLET)
        return c.get("saldo", 0.0)

    def reservar_capital(self):
        """
        Tenta pegar TODO o dinheiro para um trade.
        Retorna: Valor (float) ou 0 se estiver ocupado.
        """
        # Lógica atômica simulada: Lê -> Verifica -> Grava rápido
        carteira = self._carregar_json(ARQUIVO_WALLET)
        disponivel = carteira.get("saldo", 0.0)
        
        # Margem mínima de segurança ($10)
        if disponivel < 10: return 0.0
        
        valor_reserva = disponivel
        
        # Zera o saldo e move para "em uso"
        carteira["saldo"] = 0.0
        carteira["em_uso"] = valor_reserva
        self._salvar_json(ARQUIVO_WALLET, carteira)
        
        return valor_reserva

    def devolver_capital(self, valor_retornado):
        """Recebe o dinheiro de volta (Capital + Lucro)"""
        carteira = self._carregar_json(ARQUIVO_WALLET)
        carteira["saldo"] = valor_retornado
        carteira["em_uso"] = 0.0
        self._salvar_json(ARQUIVO_WALLET, carteira)
        return valor_retornado

    # --- LOGS E DASHBOARD ---
    def registrar_trade(self, par, lado, preco, qtd, valor_usdt, tipo, pnl_usd=0, pnl_pct=0):
        # Calcula saldo total (Livre + O que acabou de sair/entrar)
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
        """Atualiza status para o Dashboard"""
        try:
            monitor = self._carregar_json(ARQUIVO_MONITOR)
            if "moedas" not in monitor: monitor["moedas"] = []
            
            # Remove entrada antiga deste par e põe a nova
            novas = [m for m in monitor["moedas"] if m["par"] != dados_par[0]["par"]]
            novas.extend(dados_par)
            
            c = self._carregar_json(ARQUIVO_WALLET)
            total = c.get("saldo", 0) + c.get("em_uso", 0)

            monitor["moedas"] = novas
            monitor["ultima_atualizacao"] = datetime.now().strftime("%H:%M:%S")
            monitor["saldo_total"] = total
            
            self._salvar_json(ARQUIVO_MONITOR, monitor)
        except: pass
        
    def pode_enviar_alerta(self, par, timeframe):
        # Cooldown básico
        if par in self.dados:
            try:
                ultimo = datetime.fromisoformat(self.dados[par])
                if datetime.now() - ultimo < timedelta(minutes=2): return False
            except: pass
        return True

    def registrar_envio(self, par):
        self.dados[par] = datetime.now().isoformat()
        self._salvar_json(ARQUIVO_ESTADO, self.dados)