# reset_system.py
import os
import json

FILES_TO_RESET = [
    "Binance/trades_history.csv",
    "Binance/bot_state.json",
    "Binance/monitor_live.json",
    "Binance/bot_wallet.json",
    "Binance/bot_audit.log"
]

def resetar_tudo():
    print("🔥 INICIANDO O GRANDE RESET...")
    
    # 1. Apaga arquivos de histórico
    for file in FILES_TO_RESET:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"🗑️ Deletado: {file}")
            except Exception as e:
                print(f"⚠️ Erro ao deletar {file}: {e}")
        else:
            print(f"💨 Já limpo: {file}")
            
    # 2. Cria a Carteira Nova Zerada ($200)
    wallet = {
        "saldo": 0,          # Saldo Disponível
        "saldo_inicial": 0,  # Referência para cálculo de lucro total
        "em_uso": 0.0            # Quanto está preso em trades agora
    }
    
    with open("bot_wallet.json", "w") as f:
        json.dump(wallet, f, indent=4)
        
    print("\n✅ SISTEMA RESETADO!")
    print("💰 Saldo Inicial: $200.00")
    print("🚀 Pronto para o Teste de Fogo 24h.")

if __name__ == "__main__":
    resetar_tudo()