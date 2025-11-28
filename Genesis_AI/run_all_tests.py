# Genesis_AI/run_all_tests.py
import time
import os
from test_genesis_performance import PerformanceTester

def run_comprehensive_test_suite():
    print("🧪 INICIANDO SUITE DE TESTES GENESIS")
    print("="*60)
    
    # Verifica caminhos
    model_path = "cerebros/genesis_v12_final"
    data_path = "../Binance/dataset_v11_fusion.csv"
    
    if not os.path.exists(data_path):
        # Tenta caminho local se estiver rodando de dentro da pasta
        data_path = "dataset_v11_fusion.csv" 
    
    try:
        # 1. TESTE DE PERFORMANCE
        print("\n1️⃣  EXECUTANDO BACKTEST DE PERFORMANCE...")
        tester = PerformanceTester(model_path, data_path)
        perf_results = tester.run_backtest()
        tester.generate_report()
        
        # 2. AVALIAÇÃO FINAL
        print("\n🎯 VEREDITO FINAL:")
        wr = perf_results['win_rate']
        ret = perf_results['total_return']
        
        if wr > 50 and ret > 0:
            print("✅ APROVADO: A IA é lucrativa e consistente.")
        elif ret > 0:
            print("⚠️ ALERTA: Lucrativa, mas Win Rate baixo (Gestão de Risco salvou).")
        else:
            print("❌ REPROVADO: A IA perdeu dinheiro no teste.")
            
        return True

    except Exception as e:
        print(f"\n❌ Erro Crítico nos Testes: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_comprehensive_test_suite()