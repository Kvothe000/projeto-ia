# Binance/binance_connector.py
from binance.client import Client
from binance.enums import *
import pandas as pd
import config
import math

class BinanceConnector:
    def __init__(self):
        self.client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)

    def obter_saldo_usdt(self):
        """Retorna o saldo livre em USDT na carteira de Futuros"""
        try:
            account = self.client.futures_account_balance()
            for asset in account:
                if asset['asset'] == 'USDT':
                    return float(asset['balance']) # Saldo Total (Livre + Usado)
                    # Ou asset['withdrawAvailable'] para apenas o livre
            return 0.0
        except Exception as e:
            print(f"❌ Erro ao ler saldo Binance: {e}")
            return 0.0

    def buscar_candles(self, par, timeframe, limit=100):
        try:
            klines = self.client.futures_klines(symbol=par, interval=timeframe, limit=limit)
            return self._tratar_df(klines)
        except Exception as e:
            print(f"⚠️ Erro API ({par}): {e}")
            return None

    def _tratar_df(self, klines):
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.astype(float)
        return df

    def buscar_melhor_preco_book(self, par, lado):
        try:
            book = self.client.futures_order_book(symbol=par)
            if lado == "BUY": return float(book['bids'][0][0]) # Melhor Bid
            else: return float(book['asks'][0][0]) # Melhor Ask
        except: return self.obter_preco_atual(par)

    def obter_preco_atual(self, par):
        try:
            ticker = self.client.futures_symbol_ticker(symbol=par)
            return float(ticker['price'])
        except: return 0.0

    def calcular_qtd_correta(self, par, valor_usdt, preco_atual):
        try:
            info = self.client.futures_exchange_info()
            symbol_info = next((s for s in info['symbols'] if s['symbol'] == par), None)
            
            qtd_bruta = valor_usdt / preco_atual
            
            # Ajuste de precisão (stepSize)
            step_size = 0.001 # Default
            for f in symbol_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['stepSize'])
            
            precision = int(round(-math.log(step_size, 10), 0))
            return round(qtd_bruta, precision)
        except: return 0

    def cancelar_todas_ordens(self, par):
        try:
            self.client.futures_cancel_all_open_orders(symbol=par)
            return True
        except: return False

    def colocar_stop_loss(self, par, lado, qtd, preco_stop):
        try:
            self.client.futures_create_order(
                symbol=par,
                side=lado,
                type='STOP_MARKET',
                stopPrice=preco_stop,
                quantity=qtd,
                reduceOnly=True
            )
            return True
        except Exception as e:
            print(f"❌ Erro Stop Loss: {e}")
            return False

    def obter_posicao_atual(self, par):
        """
        Retorna detalhes da posição aberta na Binance.
        Retorna: dict {'qtd': float, 'preco_entrada': float, 'pnl': float, 'lado': int} ou None se zerado.
        """
        try:
            positions = self.client.futures_position_information(symbol=par)
            # A API retorna uma lista, pegamos o item correto
            for p in positions:
                if p['symbol'] == par:
                    amt = float(p['positionAmt'])
                    entry_price = float(p['entryPrice'])
                    unrealized_pnl = float(p['unRealizedProfit'])
                    
                    if amt == 0:
                        return None # Sem posição
                    
                    # 1 = Long (Qtd positiva), 2 = Short (Qtd negativa)
                    lado = 1 if amt > 0 else 2
                    
                    return {
                        'qtd': abs(amt), # Sempre positivo para cálculos
                        'qtd_real': amt, # Com sinal
                        'preco_entrada': entry_price,
                        'pnl_usd': unrealized_pnl,
                        'lado': lado
                    }
            return None
        except Exception as e:
            print(f"⚠️ Erro ao ler posição Binance: {e}")
            return None
        
    def colocar_ordem_limit(self, par, lado, qtd, preco):
        try:
            return self.client.futures_create_order(
                symbol=par, side=lado, type='LIMIT', 
                timeInForce='GTC', quantity=qtd, price=preco
            )
        except: return None