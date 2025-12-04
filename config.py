# config.py
# Arquivo de configurações do Bot Trader Professional

# =============================================================================
# CREDENCIAIS DE API E SEGURANÇA
# =============================================================================

# --- CREDENCIAIS BINANCE ---
BINANCE_API_KEY = "65E073mOBC3e8TdV1XyQvxs0ije4Aj4IXERMvZpIYOupGbCOtlczQzIhqBrvJEic"
BINANCE_API_SECRET = "Bjwxyn8Xx3Bgofz8HPPof1TV54Ur2NCE3UKzxxD1OFVzeen0fF5eE56yrMGRpk3k"

# --- CREDENCIAIS DE EMAIL ---
EMAIL_REMETENTE = "azirpaulo@gmail.com"
EMAIL_SENHA_APP = "lgdrfztfvbdmtujt"
EMAIL_DESTINATARIOS = ["azirmatheus@gmail.com"]

# =============================================================================
# GESTÃO DE CAPITAL E RISCO
# =============================================================================

# --- CONFIGURAÇÕES DE CAPITAL ---
CAPITAL_TOTAL = 100.0           # Capital base em USD (será atualizado se ler carteira)
RISCO_POR_TRADE = 1.5          # % do capital a arriscar por operação (PROFISSIONAL: 1-2%)

# --- PERFIL DE RISCO/RETORNO DIFERENCIADO ---
RISCO_RETORNO_SPOT = 2.5        # Spot: Busca movimentos maiores (Swing Trade)
RISCO_RETORNO_FUTURES = 2.0     # Futures: Operações mais rápidas (Day Trade)

# =============================================================================
# FILTROS TÉCNICOS E ESTRATÉGIA
# =============================================================================

# --- FILTROS DE ANÁLISE TÉCNICA ---
MAX_DISTANCIA_MEDIA_PCT = 7.0   # Máxima distância da EMA21 permitida (evita compras no topo)
FILTRAR_SINAIS_RUINS = False     # Bloqueia sinais com histórico ruim (WinRate < 40%)

# --- FILTROS DE MERCADO ---
PRECO_MINIMO_SATOSHIS = 0.00000150  # Preço mínimo para pares BTC
VOLUME_MINIMO_USDT = 3000000       # Volume mínimo para Futuros (3M USD)

# =============================================================================
# LISTAS DE ATIVOS
# =============================================================================

# --- PARES PARA SPOT (Acumulação) ---
PARES_SPOT = [
    'SOLBTC', 'DOGEBTC', 'LINKBTC', 'ADABTC', 'FETBTC', 
    'XRPBTC', 'AVAXBTC', 'LTCBTC', 'ARBBTC', 'ETHBTC'
]

# --- PARES PARA FUTUROS (Alavancagem) ---
PARES_FUTUROS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT', 'WLDUSDT',
    'PEPEUSDT', 'RUNEUSDT', 'APTUSDT', 'NEARUSDT'
]

# --- PARES FIXOS (Compatibilidade) ---
PARES_FIXOS = [
    'SOLBTC', 'DOGEBTC', 'LINKBTC', 'ADABTC', 'FETBTC', 
    'XRPBTC', 'AVAXBTC', 'LTCBTC', 'ARBBTC', 'ETHBTC'
]

# --- BLACKLIST (Ativos a evitar) ---
BLACKLIST = [
    'USDCBTC', 'TUSDBTC', 'USDPBTC', 'FDUSDBTC', 'DAIBTC',
    'WBTCBTC', 'ETHBTC', 'USDCUSDT', 'FDUSDUSDT'
]

# =============================================================================
# CONFIGURAÇÕES DO SISTEMA
# =============================================================================

# --- CONFIGURAÇÕES DE EMAIL ---
EMAIL_MAX_TENTATIVAS = 3        # Máximo de tentativas de envio
EMAIL_ESPERA_SEGUNDOS = 10      # Espera entre tentativas

# --- CONFIGURAÇÕES DE PERFORMANCE ---
MAX_WORKERS_PARALELO = 10       # Threads simultâneas para análise
LOOKBACK_ANALISE = "60 days ago UTC"  # Período de dados para análise
LOOKBACK_BACKTEST = "60 days ago UTC" # Período para backtesting

# =============================================================================
# CONFIGURAÇÕES AVANÇADAS (FUTURES)
# =============================================================================

# --- FUNDING RATE ---
FUNDING_RATE_ALERTA = 0.0005    # 0.05% - Alerta para funding alto
FUNDING_RATE_BONUS = -0.0001    # -0.01% - Considera funding favorável

# --- ALAVANCAGEM ---
ALAVANCAGEM_MAXIMA = 20         # Alavancagem máxima sugerida
ALAVANCAGEM_PADRAO = 10         # Alavancagem padrão para cálculos

# =============================================================================
# CONFIGURAÇÕES DE ESTRATÉGIA
# =============================================================================

# --- SAÍDA PARCIAL ---
SAIDA_PARCIAL_PERCENTUAL = 0.5  # 50% - Vende metade no meio do caminho
BREAKEVEN_TRIGGER = 0.5         # 50% - Move stop para entrada na metade

# --- TRAILING STOP ---
TRAILING_ATR_MULTIPLIER = 2.0   # 2x ATR para trailing stop

# =============================================================================
# CONFIGURAÇÕES DE LOG E MONITORAMENTO
# =============================================================================

LOG_LEVEL = "INFO"              # Nível de logging (DEBUG, INFO, WARNING, ERROR)
LOG_RETENTION_DAYS = 30         # Dias para manter logs
BACKTEST_MIN_TRADES = 5         # Mínimo de trades para considerar estatística