# Genesis_AI/run_all_tests.py
import time
from test_genesis_performance import PerformanceTester
from analyze_genesis_behavior import BehaviorAnalyzer
from stress_test_genesis import StressTester

def run_comprehensive_test_suite():
    """Executa suite completa de testes"""
    print("🧪 INICIANDO SUITE COMPLETA DE TESTES DA IA GENESIS")
    print("="*70)
    
    start_time = time.time()
    all_results = {}
    
    try:
        # 1. Teste de Performance
        print("\n1️⃣  EXECUTANDO TESTE DE PERFORMANCE...")
        perf_tester = PerformanceTester(
            "cerebros/genesis_v2_stable", 
            "dataset_v11_fusion.csv"
        )
        perf_results = perf_tester.run_backtest()
        perf_tester.generate_report()
        all_results['performance'] = perf_results
        
        # 2. Análise de Comportamento
        print("\n2️⃣  EXECUTANDO ANÁLISE DE COMPORTAMENTO...")
        feature_names = ['mom_3', 'mom_5', 'mom_10', 'vol_ratio', 'pos_canal', 
                        'trend_str', 'vol_surge', 'btc_mom', 'rel_str']
        behavior_analyzer = BehaviorAnalyzer("cerebros/genesis_v1", feature_names)
        behavior_results, decision_results = behavior_analyzer.analyze_action_patterns()
        all_results['behavior'] = behavior_results
        
        # 3. Teste de Stress
        print("\n3️⃣  EXECUTANDO TESTE DE STRESS...")
        stress_tester = StressTester("cerebros/genesis_v1")
        stress_results = stress_tester.run_market_crash_test()
        vol_results = stress_tester.run_volatility_test()
        stress_tester.generate_stress_report()
        all_results['stress'] = stress_results
        
        # 4. Avaliação Final
        print("\n🎯 AVALIAÇÃO FINAL DA IA GENESIS:")
        print("="*50)
        
        win_rate = perf_results.get('win_rate', 0)
        total_return = perf_results.get('total_return', 0)
        robustness = stress_tester._calculate_overall_robustness()
        
        if win_rate > 55 and total_return > 5 and robustness > 7:
            final_grade = "🌟 EXCELENTE - Pronta para Trading Real"
        elif win_rate > 50 and total_return > 0 and robustness > 5:
            final_grade = "✅ BOA - Pode operar com supervisão"
        else:
            final_grade = "⚠️  PRECISA DE MAIS TREINAMENTO"
        
        print(f"   Nota Final: {final_grade}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Retorno: {total_return:.2f}%") 
        print(f"   Robustez: {robustness:.1f}/10")
        
    except Exception as e:
        print(f"❌ Erro durante os testes: {e}")
        return None
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Tempo total de testes: {elapsed_time:.1f} segundos")
    print("✅ Suíte de testes concluída!")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_test_suite()
    
    if results:
        print("\n🎉 IA GENESIS TESTADA COM SUCESSO!")
        print("📁 Resultados salvos nos arquivos .png")
    else:
        print("\n❌ Falha na execução dos testes")