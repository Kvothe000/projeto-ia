import pandas as pd
import pandas_ta as ta
from binance.client import Client
from datetime import datetime
import smtplib
import socket
from email.message import EmailMessage
import schedule
import time

# --- CONFIGURAÇÃO GERAL ---
# Preencha com suas informações

# 1. Chaves da Binance (Obrigatório)
api_key = "yM3eG1C6BJnzDsDJD6JypGJUIKqfQyKC2HklRo9jnceV3dVtszGJ9gBQgmhDsyt3"
api_secret = "04H7VRYH9mbdcp7rvaJ3Q7SztihZFO6Hb6II1XoMJbXeBGFrdzLO7zelftLfmpb8"

# 2. Configurações de Email (Obrigatório)
email_remetente = "azirpaulo@gmail.com"
email_senha_app = "lgdrfztfvbdmtujt"  # Use a Senha de App gerada no Google

# 3. Lista de emails de destino
email_destinatarios = ["azirmatheus@gmail.com"]


# 4. Parâmetros da Análise
pares_fixos_para_analisar = [
    'AAVEBTC', 'SOLBTC', 'DOGEBTC', 'LINKBTC', 'ADABTC',
    'FETBTC', 'XRPBTC', 'AVAXBTC', 'TRXBTC', 'HBARBTC',
    'LTCBTC', 'XLMBTC','ARBBTC','WLDBTC','UNIBTC','ENABTC'
]

# DICIONÁRIO DE TIMEFRAMES ATUALIZADO PARA INCLUIR '1 HORA'
timeframes_base = {
    "5 minutos": Client.KLINE_INTERVAL_5MINUTE,
    "15 minutos": Client.KLINE_INTERVAL_15MINUTE,
    "1 hora": Client.KLINE_INTERVAL_1HOUR,
    "diario": Client.KLINE_INTERVAL_1DAY
}

PRECO_MINIMO_SATOSHIS = 0.00000150

# --- INICIALIZAÇÃO DO CLIENTE BINANCE ---
client = Client(api_key, api_secret)

# --- FUNÇÃO DE ENVIO DE EMAIL (ATUALIZADA COM RETRY) ---
def enviar_email(assunto, corpo_mensagem):
    MAX_TENTATIVAS = 3
    ESPERA_SEGUNDOS = 10 # Espera 10 segundos entre falhas

    # Prepara a mensagem uma única vez
    msg = EmailMessage()
    msg['Subject'] = assunto
    msg['From'] = email_remetente
    msg['To'] = ", ".join(email_destinatarios)
    msg.set_content(corpo_mensagem)

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            print(f"\nConectando ao servidor de email... (Tentativa {tentativa}/{MAX_TENTATIVAS})")
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(email_remetente, email_senha_app)
                smtp.send_message(msg)
            print("Email de alerta enviado com sucesso!")
            return  # SUCESSO! Sai da função.

        except (socket.gaierror, smtplib.SMTPException, TimeoutError) as e:
            # Pega erros de DNS (gaierror), erros de SMTP (conexão recusada, etc.) e Timeouts
            print(f"ERRO DE CONEXÃO/REDE (Tentativa {tentativa}): {e}")
            if tentativa < MAX_TENTATIVAS:
                print(f"Aguardando {ESPERA_SEGUNDOS} segundos para tentar novamente...")
                time.sleep(ESPERA_SEGUNDOS)
            else:
                print(f"ERRO FINAL: Email não enviado após {MAX_TENTATIVAS} tentativas.")
        
        except Exception as e:
            # Pega erros que NÃO devem ser retentados (ex: login/senha errada)
            print(f"ERRO CRÍTICO AO ENVIAR O EMAIL (não será retentado): {e}")
            return  # DESISTE!

    # Fim do loop, todas as tentativas de rede falharam
    print("Falha no envio do email após todas as tentativas de rede.")

# --- FUNÇÃO PARA BUSCAR PARES DINÂMICOS (sem alterações) ---
def buscar_outros_pares_btc(volume_minimo_btc):
    outros_pares_validos = []
    print("\nBuscando outros pares BTC com base nos critérios...")
    try:
        exchange_info = client.get_exchange_info()
        pares_com_alavancagem = {s['symbol'] for s in exchange_info['symbols'] if s['isMarginTradingAllowed']}
        tickers = client.get_ticker()
        for ticker in tickers:
            par = ticker['symbol']
            if par in pares_com_alavancagem:
                if par.endswith('BTC') and par not in pares_fixos_para_analisar:
                    volume_24h_btc = float(ticker['quoteVolume'])
                    preco_atual = float(ticker['lastPrice'])
                    if volume_24h_btc >= volume_minimo_btc and preco_atual >= PRECO_MINIMO_SATOSHIS:
                        outros_pares_validos.append(par)
        print(f"Encontrados {len(outros_pares_validos)} outros pares que atendem aos critérios.")
        return outros_pares_validos
    except Exception as e:
        print(f"ERRO ao buscar outros pares BTC: {e}")
        return []

# --- FUNÇÃO AUXILIAR PARA VERIFICAR RSI ---
def verificar_rsi(candle, tipo_candle):
    condicoes_encontradas = []
    if 'RSI_14' not in candle or pd.isna(candle['RSI_14']):
        return condicoes_encontradas
    rsi = candle['RSI_14']
    if rsi < 30:
        condicao_texto = (f"  -> Condição no Candle {tipo_candle.upper()}: RSI(14) [{rsi:.2f}] está abaixo de 30 (Sobrevenda).")
        condicoes_encontradas.append(condicao_texto)
        print(f"        -> ALERTA NO CANDLE {tipo_candle.upper()}: RSI < 30")
    return condicoes_encontradas

# --- FUNÇÃO DE VERIFICAÇÃO DE CONFLUÊNCIA BOLLINGER ---
def buscar_e_verificar_bollinger_superior(par, timeframe_superior_id, client_instance):
    try:
        klines_sup = client_instance.get_historical_klines(par, timeframe_superior_id, "1 day ago UTC")
        if not klines_sup:
            print(f"        -> AVISO: Não foram retornados dados para {par} no timeframe superior.")
            return False
        colunas = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                   'taker_buy_quote_asset_volume', 'ignore']
        df_sup = pd.DataFrame(klines_sup, columns=colunas)
        df_sup['low'] = pd.to_numeric(df_sup['low']).astype('float64')
        df_sup['close'] = pd.to_numeric(df_sup['close']).astype('float64')
        df_sup.ta.bbands(length=20, std=2, append=True, col_names=('BBL', 'BBM', 'BBU', 'BBB', 'BBP'))
        candle_sup_atual = df_sup.iloc[-1]
        if 'BBL' in candle_sup_atual and not pd.isna(candle_sup_atual['BBL']) and candle_sup_atual['low'] <= candle_sup_atual['BBL']:
            return True
    except Exception as e:
        print(f"        -> ERRO ao verificar confluência para {par}: {e}")
    return False

# --- FUNÇÃO PRINCIPAL DE ANÁLISE (ATUALIZADA) ---
def rodar_analise_de_alertas(pares_para_analisar, timeframe_selecionado):
    if timeframe_selecionado != 'diario':
        print("\nIniciando ciclo agendado. Aguardando 15 segundos para garantir dados do candle...")
        time.sleep(15)

    mensagens_alerta_final = []
    hora_execucao = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

    print(f"--- INICIANDO ANÁLISE DE ALERTAS ({hora_execucao}) ---")
    print(f"\nTotal de pares a serem analisados neste ciclo: {len(pares_para_analisar)}")

    timeframes_a_rodar = {}
    if timeframe_selecionado == '5':
        timeframes_a_rodar["5 minutos"] = timeframes_base["5 minutos"]
    elif timeframe_selecionado == '15':
        timeframes_a_rodar["15 minutos"] = timeframes_base["15 minutos"]
    elif timeframe_selecionado == 'diario':
        timeframes_a_rodar["diario"] = timeframes_base["diario"]
    
    for par in pares_para_analisar:
        print(f"\n--- Analisando o PAR: {par.upper()} ---")
        for timeframe_texto, timeframe_id in timeframes_a_rodar.items():
            try:
                periodo_busca = "60 days ago UTC" if timeframe_id == Client.KLINE_INTERVAL_1DAY else "3 days ago UTC"
                print(f"  Analisando o timeframe: {timeframe_texto}...")
                klines = client.get_historical_klines(par, timeframe_id, periodo_busca)

                if len(klines) < 30:
                    print(f"        -> AVISO: Dados insuficientes para análise de {par} no timeframe {timeframe_texto}. Pulando.")
                    continue

                colunas = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                df = pd.DataFrame(klines, columns=colunas)
                
                df['close'] = pd.to_numeric(df['close']).astype('float64')
                df['low'] = pd.to_numeric(df['low']).astype('float64')

                # --- CÁLCULO DE INDICADORES ---
                df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
                df.ta.bbands(length=20, std=2, append=True, col_names=('BBL', 'BBM', 'BBU', 'BBB', 'BBP'))
                df.ta.macd(fast=12, slow=26, signal=9, append=True)
                df['EMA_9'] = df.ta.ema(close=df['close'], length=9)
                df['EMA_21'] = df.ta.ema(close=df['close'], length=21)
                df.ta.stoch(k=14, d=5, smooth_k=5, append=True)
                
                candle_atual = df.iloc[-1]
                candle_anterior = df.iloc[-2]
                candle_anterior_2 = df.iloc[-3]
                
                condicoes_do_par = []
                
                required_cols_atual = ['BBL', 'STOCHk_14_5_5', 'STOCHd_14_5_5']
                if all(col in candle_atual and not pd.isna(candle_atual[col]) for col in required_cols_atual):
                    reacao_preco_ok = candle_atual['close'] > candle_atual['BBL']
                    gatilho_fundo_ok = False
                    if 'BBL' in candle_anterior and not pd.isna(candle_anterior['BBL']) and candle_anterior['low'] <= candle_anterior['BBL']:
                        gatilho_fundo_ok = True
                    elif 'BBL' in candle_anterior_2 and not pd.isna(candle_anterior_2['BBL']) and candle_anterior_2['low'] <= candle_anterior_2['BBL']:
                        gatilho_fundo_ok = True
                    if reacao_preco_ok and gatilho_fundo_ok:
                        stoch_abaixo_25 = candle_atual['STOCHk_14_5_5'] < 25 and candle_atual['STOCHd_14_5_5'] < 25
                        stoch_cruzou_acima = candle_atual['STOCHk_14_5_5'] > candle_atual['STOCHd_14_5_5']
                        if stoch_abaixo_25 and stoch_cruzou_acima:
                            condicoes_do_par.extend(["** ALERTA DE REVERSÃO PÓS-BOLLINGER **", "  -> O preço reagiu após um toque recente na Banda de Bollinger Inferior.", f"  -> Preço atual [{candle_atual['close']:.8f}] está acima da Banda [{candle_atual['BBL']:.8f}].", "  -> O Momentum de alta é confirmado pelo Estocástico (< 25 e com cruzamento K>D)."])
                            print(f"        -> ALERTA: Reversão Pós-Bollinger para {par}")

                macd_anterior, sinal_anterior = candle_anterior['MACD_12_26_9'], candle_anterior['MACDs_12_26_9']
                macd_atual, sinal_atual = candle_atual['MACD_12_26_9'], candle_atual['MACDs_12_26_9']
                if not pd.isna(macd_atual) and not pd.isna(sinal_atual) and macd_atual < 0 and sinal_atual < 0 and macd_anterior < sinal_anterior and macd_atual >= sinal_atual:
                    condicoes_do_par.extend(["** CRUZAMENTO MACD DE ALTA (ABAIXO DE ZERO) **", "  -> Ocorreu um cruzamento da linha MACD sobre a linha de Sinal em território negativo.", f"  -> Anterior: MACD [{macd_anterior:.8f}] < Sinal [{sinal_anterior:.8f}]", f"  -> Atual:    MACD [{macd_atual:.8f}] >= Sinal [{sinal_atual:.8f}]"])
                    print(f"        -> ALERTA: Cruzamento Bullish de MACD para {par}")
            
                ema9_anterior, ema21_anterior = candle_anterior['EMA_9'], candle_anterior['EMA_21']
                ema9_atual, ema21_atual = candle_atual['EMA_9'], candle_atual['EMA_21']
                if not pd.isna(ema9_atual) and not pd.isna(ema21_atual) and ema9_anterior < ema21_anterior and ema9_atual >= ema21_atual:
                    condicoes_do_par.extend(["** CRUZAMENTO DE MÉDIAS MÓVEIS DE ALTA **", "  -> A Média Móvel Exponencial de 9 períodos cruzou para cima da de 21 períodos.", f"  -> Anterior: EMA9 [{ema9_anterior:.8f}] < EMA21 [{ema21_anterior:.8f}]", f"  -> Atual:    EMA9 [{ema9_atual:.8f}] >= EMA21 [{ema21_atual:.8f}]"])
                    print(f"        -> ALERTA: Cruzamento de EMAs para {par}")
                
                if timeframe_texto != 'diario':
                    condicoes_do_par.extend(verificar_rsi(candle_anterior, "Anterior"))
                condicoes_do_par.extend(verificar_rsi(candle_atual, "Atual"))

                if timeframe_texto != 'diario':
                    gatilho_no_anterior = ('BBL' in candle_anterior and not pd.isna(candle_anterior['BBL']) and candle_anterior['low'] <= candle_anterior['BBL'])
                    gatilho_no_atual = ('BBL' in candle_atual and not pd.isna(candle_atual['BBL']) and candle_atual['low'] <= candle_atual['BBL'])
                    if gatilho_no_anterior or gatilho_no_atual:
                        gatilho_texto, candle_gatilho = ("CANDLE ANTERIOR", candle_anterior) if gatilho_no_anterior else ("CANDLE ATUAL", candle_atual)
                        print(f"    -> Gatilho Bollinger ({gatilho_texto}) encontrado. Mínima: [{candle_gatilho['low']:.8f}], Banda: [{candle_gatilho['BBL']:.8f}]")
                        timeframe_superior_id, timeframe_superior_texto = None, ""
                        if timeframe_texto == "5 minutos": timeframe_superior_id, timeframe_superior_texto = timeframes_base["15 minutos"], "15 minutos"
                        elif timeframe_texto == "15 minutos": timeframe_superior_id, timeframe_superior_texto = timeframes_base["1 hora"], "1 hora"
                        if timeframe_superior_id:
                            print(f"       -> Verificando confluência no gráfico de {timeframe_superior_texto}...")
                            if buscar_e_verificar_bollinger_superior(par, timeframe_superior_id, client):
                                condicoes_do_par.extend([f"\n** SINAL DE CONFLUÊNCIA BOLLINGER (Gatilho no {gatilho_texto}) **", f"  -> O toque na banda inferior no timeframe de {timeframe_texto} foi CONFIRMADO no timeframe de {timeframe_superior_texto}."])
                                print(f"        -> ALERTA: Confluência Bollinger CONFIRMADA para {par}!")

                if condicoes_do_par:
                    mensagens_alerta_final.append(f"ALERTA: {par} ({timeframe_texto})\n" + "\n".join(condicoes_do_par) + "\n")
            except Exception as e:
                print(f"        -> ERRO ao analisar o timeframe {timeframe_texto} para {par}: {e}")

    if mensagens_alerta_final:
        print("\n--- Compilando e enviando email de alerta... ---")
        assunto = f"Alerta Cripto: Condição(ões) encontrada(s)!"
        corpo_email = f"Análise de Alertas executada em: {hora_execucao}\n\n" + "=" * 40 + "\n\n" + "\n\n".join(mensagens_alerta_final)
        enviar_email(assunto, corpo_email)
    else:
        print("\n--- ANÁLISE CONCLUÍDA: Nenhuma condição de alerta foi encontrada. ---")

if __name__ == "__main__":
    modo_execucao_escolhido, volume_minimo_escolhido, timeframe_escolhido = '', 0, ''
    while modo_execucao_escolhido not in ['todas', 'fixas']:
        resposta_modo = input("Gostaria de rodar para TODAS as moedas ou somente para as FIXAS? (todas/fixas): ").lower()
        if resposta_modo in ['todas', 'fixas']: modo_execucao_escolhido = resposta_modo
        else: print("Opção inválida. Por favor, digite 'todas' ou 'fixas'.")
    if modo_execucao_escolhido == 'todas':
        while volume_minimo_escolhido <= 0:
            try:
                resposta_volume = float(input("Qual o volume mínimo em BTC para o filtro? (ex: 4): "))
                if resposta_volume > 0: volume_minimo_escolhido = resposta_volume
                else: print("O volume deve ser um número maior que zero.")
            except ValueError: print("Entrada inválida. Por favor, digite um número.")
    
    # --- ALTERAÇÃO CIRÚRGICA 1: Menu de tempo simplificado ---
    while timeframe_escolhido not in ['5', '15', 'diario']:
        resposta_timeframe = input("Qual timeframe rodar? (5/15/diario): ").lower()
        if resposta_timeframe in ['5', '15', 'diario']:
            timeframe_escolhido = resposta_timeframe
        else:
            print("Opção inválida. Por favor, digite '5', '15' ou 'diario'.")

    print("\n--- Preparando a lista de pares para a análise... ---")
    lista_de_pares_final = list(pares_fixos_para_analisar)
    if modo_execucao_escolhido == 'todas':
        lista_de_pares_final.extend(buscar_outros_pares_btc(volume_minimo_escolhido))
    print(f"Análise será executada com um total de {len(lista_de_pares_final)} pares.")
    
    if timeframe_escolhido == 'diario':
        print("\n--- Configuração concluída. Iniciando MODO DE EXECUÇÃO ÚNICA. ---")
        rodar_analise_de_alertas(pares_para_analisar=lista_de_pares_final, timeframe_selecionado=timeframe_escolhido)
        print("\n--- Análise diária concluída. Encerrando o programa. ---")
    else:
        # --- ALTERAÇÃO CIRÚRGICA 2: Lógica de agendamento separada ---
        print("\n--- Configuração concluída. Iniciando o MODO AGENDADO. ---")
        print(f"Timeframe a ser analisado: {timeframe_escolhido.upper()}")
        
        parametros_agendamento = {'pares_para_analisar': lista_de_pares_final, 'timeframe_selecionado': timeframe_escolhido}

        if timeframe_escolhido == '15':
            print("O programa rodará nos minutos 00, 15, 30 e 45 de cada hora.")
            schedule.every().hour.at(":00").do(rodar_analise_de_alertas, **parametros_agendamento)
            schedule.every().hour.at(":15").do(rodar_analise_de_alertas, **parametros_agendamento)
            schedule.every().hour.at(":30").do(rodar_analise_de_alertas, **parametros_agendamento)
            schedule.every().hour.at(":45").do(rodar_analise_de_alertas, **parametros_agendamento)
        
        elif timeframe_escolhido == '5':
            print("O programa rodará a cada 5 minutos.")
            # Agenda para todos os minutos múltiplos de 5
            for minute in range(0, 60, 5):
                schedule.every().hour.at(f":{minute:02d}").do(rodar_analise_de_alertas, **parametros_agendamento)

        rodar_analise_de_alertas(**parametros_agendamento)
        while True:
            schedule.run_pending()
            time.sleep(1)

