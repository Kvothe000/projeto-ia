# main.py - VERSÃO 15.0 (Professional MTF Edition)
import time
import schedule
import logging
import json
import concurrent.futures
from datetime import datetime
import config
from notifier import Notificador
from binance_connector import BinanceConnector
from indicators import Calculadora, Estrategia
from manager import GerenciadorEstado
from backtester import Backtester

# =============================================
# CONFIGURAÇÃO INICIAL
# =============================================

# Configuração de Log
logging.basicConfig(
    filename='bot_audit.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Instâncias Globais
connector = BinanceConnector()
estado = GerenciadorEstado()

# Variáveis de Estado
MODO_OPERACAO = "SPOT"
SALDO_ATUAL = config.CAPITAL_TOTAL

# Mapa de Timeframes
TIMEFRAME_MAP = {
    "5": {"id": "5m", "texto": "5min"},
    "15": {"id": "15m", "texto": "15min"}, 
    "diario": {"id": "1d", "texto": "Diário"}
}

# Hierarquia Multi-Timeframe
MTF_HIERARCHY = {
    "5m": "15m",   # 5min -> 15min (confirmação)
    "15m": "1h",   # 15min -> 1h (tendência)
    "1h": "4h",    # 1h -> 4h (direção)
    "4h": "1d",    # 4h -> diário (macro)
    "1d": "1w"     # diário -> semanal (longo prazo)
}

# =============================================
# FUNÇÕES PRINCIPAIS
# =============================================

def analisar_contexto_mercado(tf_id):
    """
    Analisa contexto BTC e força das Alts
    Retorna: (btc_ok, alts_ok, btc_price, btc_rsi)
    """
    try:
        # BTC Analysis
        df_btc = connector.buscar_candles("BTCUSDT", tf_id, lookback="30 days ago UTC")
        if df_btc is None or len(df_btc) < 50:
            return True, False, 0, 50
            
        df_btc = Calculadora.adicionar_todos(df_btc)
        candle_btc = df_btc.iloc[-1]
        
        btc_ok = candle_btc['close'] > candle_btc['EMA_21']
        btc_price = candle_btc['close']
        btc_rsi = candle_btc.get('RSI_14', 50)
        
        # Alts Strength (ETH/BTC)
        df_eth = connector.buscar_candles("ETHBTC", tf_id, lookback="15 days ago UTC")
        alts_ok = False
        if df_eth is not None and len(df_eth) > 20:
            df_eth = Calculadora.adicionar_todos(df_eth)
            alts_ok = df_eth.iloc[-1]['close'] > df_eth.iloc[-1]['EMA_21']
            
        return btc_ok, alts_ok, btc_price, btc_rsi
        
    except Exception as e:
        logging.error(f"Erro contexto mercado: {e}")
        return True, False, 0, 50

def analisar_par(par, tf_config, btc_ok, btc_rsi):
    """
    Analisa um par específico com todos os filtros incluindo MTF
    Retorna: diagnóstico completo para dashboard
    """
    # Inicializar diagnóstico
    diagnostico = {
        "par": par,
        "preco": 0,
        "regime": "N/A",
        "dist_media": 0,
        "status": "IGNORADO",
        "detalhe": "Iniciando análise",
        "sinal_email": None,
        "direcao": None,
        "win_rate": 0,
        "trades": 0,
        "saldo_historico": 0,
        "mtf_aprovado": True
    }

    # ========== FILTRO 1: ANTI-SPAM ==========
    if not estado.pode_enviar_alerta(par, tf_config['id']):
        diagnostico.update({
            "status": "EM ESPERA", 
            "detalhe": "Aguardando cooldown"
        })
        return diagnostico

    # ========== FILTRO 2: CONTEXTO BTC ==========
    if MODO_OPERACAO == "SPOT" and not btc_ok:
        diagnostico.update({
            "status": "FILTRADO", 
            "detalhe": "Mercado Bearish (BTC abaixo EMA21)"
        })
        return diagnostico

    # ========== BUSCAR DADOS MULTI-TIMEFRAME ==========
    
    # Dados do timeframe operacional
    df_operacional = connector.buscar_candles(par, tf_config['id'], lookback="60 days ago UTC")
    if df_operacional is None or len(df_operacional) < 100:
        diagnostico.update({
            "status": "SEM DADOS", 
            "detalhe": "Dados insuficientes"
        })
        return diagnostico
    
    # Dados do timeframe superior (MTF)
    tf_superior = MTF_HIERARCHY.get(tf_config['id'])
    df_superior = None
    
    if tf_superior:
        df_superior = connector.buscar_candles(par, tf_superior, lookback="100 candles ago UTC")
        if df_superior is not None and len(df_superior) > 20:
            df_superior = Calculadora.adicionar_todos(df_superior)

    # ========== CALCULAR INDICADORES ==========
    df_operacional = Calculadora.adicionar_todos(df_operacional)
    c_atual = df_operacional.iloc[-1]
    diagnostico['preco'] = float(c_atual['close'])
    
    # ========== ANÁLISE TÉCNICA COM MTF ==========
    sinais, regime, direcao = Estrategia.analisar_sinais(df_operacional, df_superior)
    
    diagnostico['regime'] = regime
    diagnostico['direcao'] = direcao
    
    # Calcular distância da média
    dist = (c_atual['close'] - c_atual['EMA_21']) / c_atual['EMA_21'] * 100
    diagnostico['dist_media'] = round(dist, 2)

    # ========== VERIFICAR SE HOUVE SINAIS ==========
    if not sinais or not direcao:
        if regime == "INDEFINIDO": 
            diagnostico.update({
                "status": "NEUTRO", 
                "detalhe": "Indicadores indefinidos"
            })
        elif abs(dist) > config.MAX_DISTANCIA_MEDIA_PCT: 
            diagnostico.update({
                "status": "ESTICADO", 
                "detalhe": f"Longe da média ({dist:.1f}%)"
            })
        else:
            # Pode ter sido bloqueado pelo MTF
            diagnostico.update({
                "status": "SEM SETUP", 
                "detalhe": "Aguardando confirmação MTF"
            })
        return diagnostico

    # ========== VERIFICAR SE MTF BLOQUEOU ==========
    if any("[MTF BLOQUEADO]" in sinal for sinal in sinais):
        diagnostico.update({
            "status": "FILTRADO MTF",
            "detalhe": "Timeframe superior contra tendência",
            "mtf_aprovado": False
        })
        return diagnostico

    # ========== FILTRO 3: MODO OPERACIONAL ==========
    if MODO_OPERACAO == "SPOT" and direcao == "SHORT":
        diagnostico.update({
            "status": "FILTRADO", 
            "detalhe": "Short não permitido em Spot"
        })
        return diagnostico

    # ========== FILTRO 4: BACKTEST ESTATÍSTICO ==========
    win_rate, total_trades, saldo_historico = Backtester.simular_sinal_no_passado(df_operacional, sinais[0])
    diagnostico.update({
        "win_rate": round(win_rate, 1),
        "trades": total_trades,
        "saldo_historico": round(saldo_historico, 1)
    })
    
    if config.FILTRAR_SINAIS_RUINS and (win_rate < 40 or saldo_historico < -5):
        diagnostico.update({
            "status": "REJEITADO", 
            "detalhe": f"Estatística ruim (WR: {win_rate:.0f}%)"
        })
        return diagnostico

    # ========== CÁLCULO DE RISCO E POSIÇÃO ==========
    dados_trade = Estrategia.calcular_posicao_e_risco(
        df_operacional.iloc[-1], SALDO_ATUAL, direcao, MODO_OPERACAO
    )
    
    if not dados_trade:
        diagnostico.update({
            "status": "ERRO CALCULO", 
            "detalhe": "Erro no cálculo de risco"
        })
        return diagnostico

    # ========== FILTRO 5: ORDEM MÍNIMA ==========
    if dados_trade['total'] < 6.0:
        diagnostico.update({
            "status": "SALDO INSUFICIENTE",
            "detalhe": f"Posição ${dados_trade['total']:.2f} < mínimo"
        })
        return diagnostico

    # ========== FILTRO 6: FUNDING RATE (FUTUROS) ==========
    aviso_funding = ""
    if MODO_OPERACAO == "FUTUROS":
        funding = connector.obter_funding_rate(par)
        if funding is not None:
            if (direcao == "LONG" and funding > 0.01) or (direcao == "SHORT" and funding < -0.01):
                aviso_funding = f"⚠️ Funding Alto: {funding*100:.3f}%"

    # ========== SINAL APROVADO! ==========
    estado.registrar_envio(par, tf_config['id'])

    # ========== MONTAR ALERTA DE EMAIL ==========
    emoji_dir = "🟢" if direcao == "LONG" else "🔴"
    emoji_conf = "💎" if win_rate > 60 else "🎯" if win_rate > 50 else "🎲"
    
    # Info MTF para email
    info_mtf = ""
    if df_superior is not None:
        trend_sup = "Bull" if df_superior.iloc[-1]['close'] > df_superior.iloc[-1]['EMA_21'] else "Bear"
        info_mtf = f" | MTF: {trend_sup}"

    # Calcular preço de breakeven
    if direcao == "LONG":
        be_price = c_atual['close'] + (dados_trade['tp'] - c_atual['close']) * 0.5
    else:
        be_price = c_atual['close'] - (c_atual['close'] - dados_trade['tp']) * 0.5

    # Montar alerta
    alerta = [
        f"{emoji_dir} {direcao} {par} | {tf_config['texto']}{info_mtf}",
        f"📊 {sinais[0].replace('[MTF BLOQUEADO]', '').strip()}",
        f"📈 Regime: {regime} | Preço: {c_atual['close']:.8f}",
        f"{emoji_conf} Estatísticas: {win_rate:.1f}% Win ({total_trades} trades) | Hist: {saldo_historico:.1f}%",
        f"🛑 Stop: {dados_trade['stop']:.8f} (-{dados_trade['stop_perc']:.2f}%)",
        f"🎯 Alvo: {dados_trade['tp']:.8f}",
        f"💰 Posição: {dados_trade['qtd']:.4f} {'moedas' if MODO_OPERACAO == 'SPOT' else 'contratos'}",
        f"💵 Valor Total: ${dados_trade['total']:.2f}",
    ]
    
    # Info específica para Futuros
    if MODO_OPERACAO == "FUTUROS":
        alerta.append(f"🔧 Alavancagem: {dados_trade['alavancagem']}x (Margem Isolada)")
        alerta.append(f"💳 Margem Necessária: ${(dados_trade['total']/dados_trade['alavancagem']):.2f}")
    
    # Gestão profissional
    alerta.append(f"🛡️ Mover Stop para Entrada em: {be_price:.8f}")
    alerta.append(f"🔄 Trailing Stop: {dados_trade['trailing']:.8f}")
    
    # Alertas adicionais
    if aviso_funding: 
        alerta.append(aviso_funding)
    if btc_rsi > 70: 
        alerta.append(f"⚠️ BTC Sobrecomprado (RSI: {btc_rsi:.1f})")
    elif btc_rsi < 30: 
        alerta.append(f"⚠️ BTC Sobrevendido (RSI: {btc_rsi:.1f})")

    # Atualizar diagnóstico final
    diagnostico.update({
        "status": "SINAL ENCONTRADO",
        "detalhe": f"{sinais[0].replace('[MTF BLOQUEADO]', '').strip()} ({direcao})",
        "sinal_email": "\n".join(alerta)
    })

    logging.info(f"Sinal {direcao} {par} APROVADO | WR: {win_rate:.1f}% | MTF: {diagnostico['mtf_aprovado']}")
    return diagnostico

def executar_scanner(pares, tf_key):
    """
    Função principal do scanner com dashboard
    """
    global SALDO_ATUAL
    
    hora_atual = datetime.now().strftime('%H:%M:%S')
    print(f"\n🎯 SCANNER {MODO_OPERACAO} v15.0 MTF ({hora_atual})")
    print("=" * 50)
    
    # ========== ATUALIZAR SALDO ==========
    saldo_real = connector.ler_saldo_atual(MODO_OPERACAO)
    if saldo_real:
        SALDO_ATUAL = saldo_real
        print(f"💰 Banca Real: ${SALDO_ATUAL:.2f}")
    else:
        print(f"💰 Banca Fixa: ${SALDO_ATUAL:.2f}")
    
    # ========== CONFIGURAÇÃO TIMEFRAME ==========
    tf_config = TIMEFRAME_MAP[tf_key]
    estado.limpar_antigos()
    
    # ========== ANÁLISE DE CONTEXTO ==========
    btc_ok, alts_ok, btc_price, btc_rsi = analisar_contexto_mercado(tf_config['id'])
    
    # Determinar status do mercado
    if btc_ok and alts_ok:
        status = "BULLISH (BTC e Alts fortes)"
    elif btc_ok and not alts_ok:
        status = "BTC DOMINANCE (Alts fracas)"
    else:
        status = "BEARISH (Cuidado)"
    
    tf_superior = MTF_HIERARCHY.get(tf_config['id'], "N/A")
    
    print(f"📊 {status}")
    print(f"₿  BTC: ${btc_price:.0f} | RSI: {btc_rsi:.1f}")
    print(f"🎯 MTF: {tf_config['id']} → {tf_superior}")
    print(f"🔍 Analisando {len(pares)} pares...")
    
    # ========== PREPARAR DASHBOARD ==========
    snapshot_dashboard = {
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "modo": MODO_OPERACAO,
        "status": status,
        "btc_price": btc_price,
        "btc_rsi": btc_rsi,
        "saldo_atual": SALDO_ATUAL,
        "mtf_config": f"{tf_config['id']}→{tf_superior}",
        "total_pares": len(pares),
        "analise_ativos": []
    }
    
    # ========== TRAVA DE SEGURANÇA ==========
    operar = True
    if not btc_ok and MODO_OPERACAO == "SPOT":
        print("⛔ SPOT PAUSADO - Mercado Bearish")
        operar = False
        snapshot_dashboard['status_mercado'] = "⛔ PAUSADO (Bear Market)"

    # ========== EXECUÇÃO PARALELA ==========
    relatorio_email = []
    estatisticas = {
        "sinais_encontrados": 0,
        "filtros_mtf": 0,
        "filtros_backtest": 0,
        "filtros_gerais": 0
    }
    
    if operar:
        print("🔄 Iniciando análise paralela...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(analisar_par, par, tf_config, btc_ok, btc_rsi): par 
                for par in pares
            }
            
            for future in concurrent.futures.as_completed(futures):
                par = futures[future]
                try:
                    diagnostico = future.result()
                    snapshot_dashboard['analise_ativos'].append(diagnostico)
                    
                    # Contabilizar estatísticas
                    if diagnostico['sinal_email']:
                        relatorio_email.append(diagnostico['sinal_email'])
                        estatisticas['sinais_encontrados'] += 1
                        print(f"✅ {par}")
                    else:
                        status_par = diagnostico['status']
                        if "MTF" in status_par:
                            estatisticas['filtros_mtf'] += 1
                        elif "REJEITADO" in status_par:
                            estatisticas['filtros_backtest'] += 1
                        elif "FILTRADO" in status_par:
                            estatisticas['filtros_gerais'] += 1
                            
                        print(".", end="", flush=True)
                        
                except Exception as e:
                    logging.error(f"Erro em {par}: {e}")
                    print(f"❌ {par}")

    # ========== SALVAR DASHBOARD ==========
    snapshot_dashboard['estatisticas'] = estatisticas
    
    try:
        nome_arquivo = f"dashboard_{MODO_OPERACAO.lower()}.json"
        with open(nome_arquivo, "w", encoding='utf-8') as f:
            json.dump(snapshot_dashboard, f, indent=2, ensure_ascii=False)
        print(f"📊 Dashboard salvo: {nome_arquivo}")
    except Exception as e:
        print(f"❌ Erro salvando dashboard: {e}")

    # ========== ENVIAR RELATÓRIO POR EMAIL ==========
    if relatorio_email:
        assunto = f"🚀 {estatisticas['sinais_encontrados']} SINAIS {MODO_OPERACAO} - {hora_atual}"
        
        # Contexto do mercado
        contexto = f"💰 Saldo: ${SALDO_ATUAL:.2f} | {status}\n"
        contexto += f"₿  BTC: ${btc_price:.0f} | RSI: {btc_rsi:.1f}\n"
        contexto += f"🎯 MTF: {tf_config['id']} → {tf_superior}\n"
        contexto += f"📊 Estatísticas: {estatisticas['sinais_encontrados']}/{len(pares)} sinais\n"
        contexto += f"🚫 Filtros: MTF({estatisticas['filtros_mtf']}) | Backtest({estatisticas['filtros_backtest']}) | Geral({estatisticas['filtros_gerais']})\n\n"
        
        corpo = contexto + "\n\n".join(relatorio_email)
        
        Notificador.enviar_email(assunto, corpo)
        print(f"\n📧 EMAIL ENVIADO: {estatisticas['sinais_encontrados']} alertas MTF")
    else:
        print("\n📭 NENHUM SINAL: Filtros MTF rigorosos aplicados")

    # ========== RELATÓRIO FINAL ==========
    print(f"\n📈 RESUMO DA EXECUÇÃO:")
    print(f"   ✅ Sinais Encontrados: {estatisticas['sinais_encontrados']}")
    print(f"   🚫 Filtrados por MTF: {estatisticas['filtros_mtf']}")
    print(f"   📊 Filtrados por Backtest: {estatisticas['filtros_backtest']}")
    print(f"   ⚠️  Filtrados Gerais: {estatisticas['filtros_gerais']}")
    print(f"   📋 Total Analisados: {len(pares)}")

# =============================================
# EXECUÇÃO PRINCIPAL
# =============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 BOT TRADER PROFESSIONAL v15.0 (MTF EDITION)")
    print("=" * 60)
    
    # ========== CONFIGURAÇÃO INICIAL ==========
    
    # Seleção do Modo
    while True:
        modo = input("\n💱 Modo de Operação (SPOT/FUTUROS): ").upper()
        if modo in ["SPOT", "FUTUROS"]:
            MODO_OPERACAO = modo
            break
        print("❌ Por favor, digite SPOT ou FUTUROS")

    # Seleção de Pares
    if MODO_OPERACAO == "SPOT":
        pares_base = list(config.PARES_SPOT)
        quote_asset = "BTC"
        print("🪙 Modo: SPOT (Acumulação de BTC)")
    else:
        pares_base = list(config.PARES_FUTUROS)
        quote_asset = "USDT" 
        print("⚡ Modo: FUTUROS (Alavancagem USDT)")

    # Opção de Lista Dinâmica
    if input("\n🔍 Buscar pares dinâmicos? (s/n): ").lower() == 's':
        volume_minimo = 3000000 if MODO_OPERACAO == "FUTUROS" else 2
        pares_dinamicos = connector.buscar_pares_dinamicos(volume_minimo, quote_asset)
        pares_base.extend(pares_dinamicos)
        pares_base = list(set(pares_base))
        print(f"📈 Adicionados {len(pares_dinamicos)} pares dinâmicos")

    pares_finais = pares_base

    # Seleção do Timeframe
    while True:
        tf_input = input("\n⏰ Timeframe (5/15/diario): ").lower()
        if tf_input in TIMEFRAME_MAP:
            break
        print("❌ Por favor, digite 5, 15 ou diario")

    tf_config = TIMEFRAME_MAP[tf_input]
    tf_superior = MTF_HIERARCHY.get(tf_config['id'], "Nenhum")

    # ========== RESUMO DA CONFIGURAÇÃO ==========
    print(f"\n{' CONFIGURAÇÃO FINAL ':=^50}")
    print(f"   🎯 Modo: {MODO_OPERACAO}")
    print(f"   📊 Pares: {len(pares_finais)}")
    print(f"   ⏰ Timeframe: {tf_config['texto']}")
    print(f"   🎯 MTF: {tf_config['id']} → {tf_superior}")
    print(f"   💰 Capital: ${SALDO_ATUAL:.2f}")
    print(f"   🛡️  Risco/Trade: {config.RISCO_POR_TRADE}%")
    print(f"   📈 Pares: {config.PARES_FUTUROS if MODO_OPERACAO=='FUTUROS' else config.PARES_SPOT}")
    print(f"   🔧 Threads: 8 paralelas")
    print(f"   📊 Dashboard: dashboard_{MODO_OPERACAO.lower()}.json")
    print("=" * 50)

    # ========== EXECUÇÃO DO SCANNER ==========
    
    # Execução imediata
    executar_scanner(pares_finais, tf_input)
    
    # Agendamento para timeframes intraday
    if tf_input != 'diario':
        intervalo = int(tf_input)
        schedule.every(intervalo).minutes.do(executar_scanner, pares=pares_finais, tf_key=tf_input)
        print(f"\n⏰ Scanner agendado a cada {intervalo} minutos...")
    else:
        schedule.every(24).hours.do(executar_scanner, pares=pares_finais, tf_key=tf_input)
        print(f"\n⏰ Scanner agendado diariamente...")
    
    print("📊 Dashboard atualizando em tempo real...")
    print("🛑 Pressione Ctrl+C para parar o bot")
    print("=" * 50)

    # ========== LOOP PRINCIPAL ==========
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # Verifica a cada 30 segundos
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Bot interrompido pelo usuário")
        print("📊 Dados finais salvos nos dashboards")
        logging.info("Bot finalizado pelo usuário")