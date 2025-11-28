# Binance/binance_connector.py - VERSÃO CORRIGIDA (INT TIMESTAMP)
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import config
import time

class BinanceConnector:
    def __init__(self):
        try:
            self.client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
            print("✅ Conexão Binance OK")
        except Exception as e:
            print(f"❌ Erro Conexão: {e}")
            raise

    def _tratar_df(self, klines):
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qav', 'nt', 'taker_buy_base', 'tbq', 'ig']
        df = pd.DataFrame(klines, columns=cols)
        
        # --- CORREÇÃO CRÍTICA: TIMESTAMP COMO INTEIRO ---
        df['timestamp'] = df['timestamp'].astype('int64')
        # ------------------------------------------------
        
        numericos = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']
        df[numericos] = df[numericos].apply(pd.to_numeric, errors='coerce')
        
        # CVD
        df['delta_volume'] = df['taker_buy_base'] - (df['volume'] - df['taker_buy_base'])
        df['CVD'] = df['delta_volume'].cumsum()
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'CVD']]

    def buscar_candles(self, par, timeframe, mercado="FUTUROS", limit=1000):
        try:
            if mercado == "FUTUROS":
                klines = self.client.futures_klines(symbol=par, interval=timeframe, limit=limit)
            else:
                klines = self.client.get_klines(symbol=par, interval=timeframe, limit=limit)
            return self._tratar_df(klines)
        except: return None

    def buscar_melhor_preco_book(self, par, lado):
        try:
            book = self.client.futures_order_book(symbol=par, limit=5)
            if lado == 'BUY': return float(book['bids'][0][0])
            else: return float(book['asks'][0][0])
        except: return 0

    def obter_precisao_moeda(self, par):
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == par:
                    step = float(next(f['stepSize'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE'))
                    tick = float(next(f['tickSize'] for f in s['filters'] if f['filterType'] == 'PRICE_FILTER'))
                    return step, tick
            return None, None
        except: return None, None

    def calcular_qtd_correta(self, par, valor_usdt, preco):
        step, _ = self.obter_precisao_moeda(par)
        if not step: return 0
        qtd = (valor_usdt / preco) // step * step
        casas = len(str(step).split('.')[1]) if '.' in str(step) else 0
        return float(f"{qtd:.{casas}f}")

    def colocar_ordem_limit(self, par, lado, qtd, preco, alavancagem=5):
        try:
            self.client.futures_change_leverage(symbol=par, leverage=alavancagem)
            print(f"🕒 Ordem LIMIT {lado} em {par} a {preco}...")
            return self.client.futures_create_order(
                symbol=par, side=lado, type='LIMIT', timeInForce='GTC',
                quantity=qtd, price=str(preco)
            )
        except Exception as e:
            print(f"❌ Erro Ordem: {e}")
            return None

    def colocar_stop_loss(self, par, lado_stop, qtd, preco_stop):
        try:
            print(f"🛡️ STOP LOSS {lado_stop} a {preco_stop}...")
            return self.client.futures_create_order(
                symbol=par, side=lado_stop, type='STOP_MARKET',
                stopPrice=str(preco_stop), closePosition=True
            )
        except: return None
            
    def cancelar_todas_ordens(self, par):
        try:
            self.client.futures_cancel_all_open_orders(symbol=par)
        except: pass