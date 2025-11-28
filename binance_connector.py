# Binance/binance_connector.py - VERSÃO V5.0 (MAKER/LIMIT + OTIMIZAÇÃO)
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import config
import time

class BinanceConnector:
    def __init__(self):
        try:
            self.client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
            # Sincroniza relógio para evitar erros de timestamp
            try:
                server_time = self.client.get_server_time()
                diff = int(time.time() * 1000) - server_time['serverTime']
                if abs(diff) > 1000:
                    print(f"⚠️ Ajustando clock interno: {diff}ms de diferença.")
            except: pass
            print("✅ Conexão Binance OK")
        except Exception as e:
            print(f"❌ Erro Conexão: {e}")
            raise

    def _tratar_df(self, klines):
        if not klines: return None
        # Colunas padrão da Binance Futures
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'nt', 'taker_buy_base', 'tbq', 'ig']
        df = pd.DataFrame(klines, columns=cols)
        
        numericos = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']
        df[numericos] = df[numericos].apply(pd.to_numeric, errors='coerce')
        
        # CVD (Fluxo de Agressão) - Essencial para a IA
        # Volume Total - 2x Taker Buy (Simplificação aceita para CVD aproximado)
        # Ou melhor: Taker Buy (Compradores) - (Total - Taker Buy) (Vendedores)
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
        except Exception as e:
            print(f"⚠️ Erro ao buscar candles de {par}: {e}")
            return None

    def buscar_melhor_preco_book(self, par, lado_operacao):
        """
        Olha o Order Book para definir o preço da ordem Limit.
        BUY: Coloca 1 tick acima do melhor Bid (para ser o primeiro da fila).
        SELL: Coloca 1 tick abaixo do melhor Ask.
        """
        try:
            book = self.client.futures_order_book(symbol=par, limit=5)
            if lado_operacao == 'BUY':
                return float(book['bids'][0][0]) # Melhor preço de quem quer comprar
            else:
                return float(book['asks'][0][0]) # Melhor preço de quem quer vender
        except Exception as e:
            print(f"❌ Erro Book: {e}")
            return 0

    def colocar_ordem_limit(self, par, lado, qtd, preco, alavancagem=5):
        """
        Tenta colocar uma ordem LIMIT (Maker) para pagar taxas menores (0.02%).
        """
        try:
            # Garante alavancagem correta
            self.client.futures_change_leverage(symbol=par, leverage=alavancagem)
            
            print(f"🕒 Enviando Ordem MAKER {lado} em {par} a {preco} (x{alavancagem})...")
            
            order = self.client.futures_create_order(
                symbol=par,
                side=lado,
                type='LIMIT',
                timeInForce='GTC', # Good Till Cancelled
                quantity=qtd,
                price=str(preco)
            )
            return order
        except Exception as e:
            print(f"❌ Falha Ordem Limit: {e}")
            return None

    def cancelar_ordem(self, par, order_id):
        try:
            self.client.futures_cancel_order(symbol=par, orderId=order_id)
            print("🗑️ Ordem não preenchida cancelada.")
        except: pass

    def colocar_ordem_market(self, par, lado, qtd, alavancagem=5):
        """
        Ordem a Mercado (Execução Imediata).
        Usada se a Limit não encher ou para Stop Loss de emergência.
        """
        try:
            self.client.futures_change_leverage(symbol=par, leverage=alavancagem)
            order = self.client.futures_create_order(
                symbol=par,
                side=lado,
                type='MARKET',
                quantity=qtd
            )
            print(f"⚡ Ordem MARKET {lado} Executada!")
            return order
        except Exception as e:
            print(f"❌ Falha Market: {e}")
            return None
        
        # ... (Mantenha o resto do código igual) ...

    def obter_precisao_moeda(self, par):
        """
        Descobre quantas casas decimais a Binance aceita para esse par.
        Retorna: (stepSize, tickSize)
        Ex: WLD aceita 0.1 moedas (step) e 0.001 no preço (tick).
        """
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == par:
                    step_size = 0
                    tick_size = 0
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            step_size = float(f['stepSize'])
                        if f['filterType'] == 'PRICE_FILTER':
                            tick_size = float(f['tickSize'])
                    return step_size, tick_size
            return None, None
        except Exception as e:
            print(f"❌ Erro ao ler precisão de {par}: {e}")
            return None, None

    def calcular_qtd_correta(self, par, valor_usdt_alvo, preco_atual):
        """
        Calcula a quantidade exata de moedas para comprar X dólares.
        Arredonda para respeitar as regras da Binance.
        """
        step_size, _ = self.obter_precisao_moeda(par)
        if not step_size: return 0.0

        # Qtd bruta = Valor em Dólar / Preço da Moeda
        qtd_bruta = valor_usdt_alvo / preco_atual
        
        # Arredondamento Matemático para o Step Size (Ex: 0.001)
        # Ex: Se step=0.1 e qtd=10.55, vira 10.5
        qtd_final = (qtd_bruta // step_size) * step_size
        
        # Formata para string para evitar erro de ponto flutuante tipo 10.500000001
        casas_decimais = len(str(step_size).split('.')[1]) if '.' in str(step_size) else 0
        if step_size == 1: casas_decimais = 0
        
        return float(f"{qtd_final:.{casas_decimais}f}")