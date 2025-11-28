# Binance/manager.py (VERSÃO VISUAL HISTORY)
import json
import os
import pandas as pd
from datetime import datetime, timedelta

ARQUIVO_ESTADO = "bot_state.json"
ARQUIVO_TRADES = "trades_history.csv"
ARQUIVO_ANALISES = "analysis_history.csv" # <--- NOVO
ARQUIVO_MONITOR = "monitor_live.json"

class GerenciadorEstado:
    def __init__(self):
        self.dados = self._carregar_estado()
        self._inicializar_csvs()

    def _carregar_estado(self):
        if os.path.exists(ARQUIVO_ESTADO):
            try:
                with open(ARQUIVO_ESTADO, 'r') as f:
                    return json.load(f)
            except: return {}
        return {}

    def _inicializar_csvs(self):
        # CSV de Trades
        if not os.path.exists(ARQUIVO_TRADES):
            pd.DataFrame(columns=[
                "data", "par", "lado", "preco", "qtd", "valor_usdt", "tipo", "resultado"
            ]).to_csv(ARQUIVO_TRADES, index=False)
            
        # CSV de Análises (NOVO)
        if not os.path.exists(ARQUIVO_ANALISES):
            pd.DataFrame(columns=[
                "data", "par", "preco", "adx", "sinal", "confianca"
            ]).to_csv(ARQUIVO_ANALISES, index=False)

    def _salvar_estado(self):
        with open(ARQUIVO_ESTADO, 'w') as f:
            json.dump(self.dados, f)

    def pode_enviar_alerta(self, par, timeframe_id):
        if par in self.dados:
            try:
                ultimo = datetime.fromisoformat(self.dados[par])
                if datetime.now() - ultimo < timedelta(minutes=3):
                    return False
            except: pass
        return True

    def registrar_envio(self, par):
        self.dados[par] = datetime.now().isoformat()
        self._salvar_estado()

    def registrar_trade(self, par, lado, preco, qtd, valor_usdt, tipo="LIMIT"):
        novo = {
            "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "par": par, "lado": lado, "preco": preco, "qtd": qtd, 
            "valor_usdt": valor_usdt, "tipo": tipo, "resultado": "ABERTO"
        }
        self._append_csv(ARQUIVO_TRADES, novo)

    def registrar_analise(self, par, preco, adx, sinal, confianca):
        """Salva o pensamento da IA (para gráficos)"""
        novo = {
            "data": datetime.now().strftime("%H:%M:%S"), # Só hora para ficar leve
            "timestamp": datetime.now().timestamp(),     # Para ordenar
            "par": par, "preco": preco, "adx": adx, 
            "sinal": sinal, "confianca": confianca
        }
        self._append_csv(ARQUIVO_ANALISES, novo)

    def _append_csv(self, arquivo, linha_dict):
        try:
            df_novo = pd.DataFrame([linha_dict])
            # Modo 'a' (append) com header apenas se arquivo não existir
            header = not os.path.exists(arquivo)
            df_novo.to_csv(arquivo, mode='a', header=header, index=False)
        except Exception as e:
            print(f"❌ Erro CSV {arquivo}: {e}")

    def atualizar_monitor(self, lista_status):
        dados = {"ultima_atualizacao": datetime.now().strftime("%H:%M:%S"), "moedas": lista_status}
        try:
            with open(ARQUIVO_MONITOR, 'w') as f:
                json.dump(dados, f)
        except: pass