# Binance/scanner.py (VERSÃO DINÂMICA V2.0)
import pandas as pd
import time
from binance_connector import BinanceConnector

class ScannerCrypto:
    def __init__(self):
        self.connector = BinanceConnector()
        print("📡 Radar Dinâmico Iniciado...")

    def buscar_top_moedas_explosivas(self):
        """
        Pede à Binance o resumo de TODAS as moedas e filtra as melhores de HOJE.
        Critérios:
        1. Alto Volume (para ter liquidez e não ficarmos presos)
        2. Alta Variação (para termos chance de lucro)
        """
        print("\n🛰️ Satélite: Escaneando o mercado inteiro (24h)...")
        try:
            # Pega dados de 24h de TODOS os pares de Futuros
            tickers = self.connector.client.futures_ticker()
            df = pd.DataFrame(tickers)
            
            # Filtros Básicos
            # 1. Apenas pares USDT
            df = df[df['symbol'].str.endswith('USDT')]
            # 2. Converte colunas para números
            df['quoteVolume'] = pd.to_numeric(df['quoteVolume']) # Volume em $
            df['priceChangePercent'] = pd.to_numeric(df['priceChangePercent'])
            
            # 3. Filtro de Liquidez: Volume > 50 Milhões de Dólares (Evita "shitcoins" presas)
            # Isso garante que não entramos em moedas mortas
            df_liquidas = df[df['quoteVolume'] > 50_000_000].copy()
            
            # 4. Ordenar pela "Agitação" (Variação absoluta)
            # Queremos o que está se mexendo muito (seja subindo ou caindo)
            df_liquidas['agito'] = df_liquidas['priceChangePercent'].abs()
            
            # Pega as Top 15 mais agitadas
            top_moedas = df_liquidas.sort_values('agito', ascending=False).head(15)
            
            lista_final = top_moedas['symbol'].tolist()
            
            print(f"✅ Radar atualizado! {len(lista_final)} moedas na mira.")
            print(f"🔥 Destaques: {lista_final[:5]}")
            
            return lista_final
            
        except Exception as e:
            print(f"❌ Erro no Radar Satélite: {e}")
            # Lista de emergência se a API falhar
            return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'WLDUSDT', '1000PEPEUSDT']

    def analisar_mercado(self):
        # 1. Atualiza a lista dinamicamente
        moedas_alvo = self.buscar_top_moedas_explosivas()
        
        ranking = []
        print(f"\n🔍 Analisando profundamente as {len(moedas_alvo)} eleitas...")
        
        for par in moedas_alvo:
            try:
                # Busca dados detalhados (Candles 15m) para ver o "agora"
                df = self.connector.buscar_candles(par, '15m', mercado="FUTUROS", limit=50)
                
                if df is None or len(df) < 30: continue

                # Cálculos de Volatilidade Recente (Últimos 75 min)
                # A variação de 24h pode ser velha. Aqui vemos quem está mexendo AGORA.
                preco_atual = df.iloc[-1]['close']
                df['amplitude'] = (df['high'] - df['low']) / df['low'] * 100
                volatilidade_recente = df['amplitude'].tail(5).mean()
                
                # Volume Power
                vol_media = df['volume'].rolling(20).mean().iloc[-1]
                vol_atual = df.iloc[-1]['volume']
                vol_power = vol_atual / vol_media if vol_media > 0 else 0
                
                # Pontuação Final
                score = volatilidade_recente * vol_power 

                ranking.append({
                    'par': par,
                    'preco': preco_atual,
                    'volatilidade_pct': volatilidade_recente,
                    'explosao_volume': vol_power,
                    'score': score
                })
                
                time.sleep(0.1) # Respeita limite da API
                
            except Exception as e:
                pass

        # Ordena do Melhor para o Pior
        df_rank = pd.DataFrame(ranking)
        if not df_rank.empty:
            df_rank = df_rank.sort_values('score', ascending=False)
        
        return df_rank

    def mostrar_top_oportunidades(self):
        df = self.analisar_mercado()
        
        if df is None or df.empty:
            print("❌ Nenhuma oportunidade encontrada.")
            return []

        print("\n🏆 TOP 5 CAMPEÃS DO MOMENTO:")
        print(f"{'PAR':<12} | {'VOLATIL.(15m)':<15} | {'VOL. POWER':<10}")
        print("-" * 45)
        
        # Pega as TOP 5
        top5 = df.head(5)
        for _, row in top5.iterrows():
            vol_str = f"{row['volatilidade_pct']:.2f}%"
            pow_str = f"{row['explosao_volume']:.1f}x"
            print(f"{row['par']:<12} | {vol_str:<15} | {pow_str:<10}")
            
        # Retorna a lista de nomes (Ex: ['TRADOORUSDT', 'MUSDT', ...])
        lista_moedas = top5['par'].tolist()
        print(f"\n👉 Alvos Carregados: {lista_moedas}")
        return lista_moedas

if __name__ == "__main__":
    scanner = ScannerCrypto()
    scanner.mostrar_top_oportunidades()