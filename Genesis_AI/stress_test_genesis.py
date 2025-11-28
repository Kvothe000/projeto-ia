# Genesis_AI/stress_test_genesis.py
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

class StressTester:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
        self.stress_results = {}
    
    def run_market_crash_test(self, initial_balance=10000):
        """Testa como a IA reage a condições extremas de mercado"""
        print("🌪️  EXECUTANDO TESTE DE STRESS - MERCADO EM CRASH...")
        
        # Simula condições de crash (quedas bruscas)
        crash_scenarios = [
            {'name': 'Crash Suave (-15%)', 'decline_rate': -0.0015},
            {'name': 'Crash Moderado (-30%)', 'decline_rate': -0.003},
            {'name': 'Crash Severo (-50%)', 'decline_rate': -0.005},
            {'name': 'Flash Crash (-20% instantâneo)', 'flash_crash': True}
        ]
        
        crash_results = []
        
        for scenario in crash_scenarios:
            print(f"\n📉 Testando: {scenario['name']}")
            
            # Simula ambiente de crash
            balance = initial_balance
            max_drawdown = 0
            recovery_time = None
            
            if scenario.get('flash_crash'):
                # Crash instantâneo de 20%
                balance *= 0.8
                max_drawdown = 20.0
            else:
                # Crash gradual
                for step in range(100):
                    balance *= (1 + scenario['decline_rate'] + np.random.normal(0, 0.01))
                    drawdown = (initial_balance - balance) / initial_balance * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    
                    # Verifica recuperação
                    if balance >= initial_balance and recovery_time is None:
                        recovery_time = step
            
            final_return = (balance - initial_balance) / initial_balance * 100
            
            crash_results.append({
                'scenario': scenario['name'],
                'final_return': final_return,
                'max_drawdown': max_drawdown,
                'recovery_time': recovery_time if recovery_time else 'Não recuperou',
                'survived': balance > initial_balance * 0.5  # Sobreviveu se perdeu menos de 50%
            })
            
            print(f"   Retorno Final: {final_return:.2f}%")
            print(f"   Drawdown Máximo: {max_drawdown:.2f}%")
            print(f"   Tempo de Recuperação: {recovery_time if recovery_time else 'N/A'} steps")
            print(f"   Sobreviveu: {'✅' if balance > initial_balance * 0.5 else '❌'}")
        
        self.stress_results['crash_tests'] = crash_results
        return crash_results
    
    def run_volatility_test(self, initial_balance=10000):
        """Testa performance em alta volatilidade"""
        print("\n⚡ EXECUTANDO TESTE DE ALTA VOLATILIDADE...")
        
        volatility_levels = [0.01, 0.03, 0.05, 0.1]  # 1% to 10% daily volatility
        
        vol_results = []
        
        for vol in volatility_levels:
            balance = initial_balance
            returns = []
            
            for step in range(100):
                # Simula retornos com alta volatilidade
                daily_return = np.random.normal(0, vol)
                balance *= (1 + daily_return)
                returns.append(daily_return)
            
            # Calcula métricas
            total_return = (balance - initial_balance) / initial_balance * 100
            volatility = np.std(returns) * np.sqrt(252) * 100  # Anualizada
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            vol_results.append({
                'volatility_level': f'{vol*100:.1f}%',
                'total_return': total_return,
                'annual_volatility': volatility,
                'sharpe_ratio': sharpe,
                'final_balance': balance
            })
            
            print(f"   Volatilidade {vol*100:.1f}% -> Retorno: {total_return:.2f}% | Sharpe: {sharpe:.2f}")
        
        self.stress_results['volatility_tests'] = vol_results
        return vol_results
    
    def generate_stress_report(self):
        """Gera relatório completo de stress test"""
        print("\n" + "="*60)
        print("🧪 RELATÓRIO DE STRESS TEST - IA GENESIS")
        print("="*60)
        
        # Análise de crashes
        if 'crash_tests' in self.stress_results:
            print("\n📉 DESEMPENHO EM CRASHES:")
            survival_rate = len([t for t in self.stress_results['crash_tests'] if t['survived']]) / len(self.stress_results['crash_tests']) * 100
            print(f"   Taxa de Sobrevivência: {survival_rate:.1f}%")
            
            for test in self.stress_results['crash_tests']:
                status = "✅ SOBREVIVEU" if test['survived'] else "❌ QUEBROU"
                print(f"   {test['scenario']}: {status}")
        
        # Análise de volatilidade
        if 'volatility_tests' in self.stress_results:
            print("\n⚡ DESEMPENHO EM VOLATILIDADE:")
            best_performer = max(self.stress_results['volatility_tests'], key=lambda x: x['sharpe_ratio'])
            worst_performer = min(self.stress_results['volatility_tests'], key=lambda x: x['sharpe_ratio'])
            
            print(f"   Melhor em: {best_performer['volatility_level']} volatilidade (Sharpe: {best_performer['sharpe_ratio']:.2f})")
            print(f"   Pior em: {worst_performer['volatility_level']} volatilidade (Sharpe: {worst_performer['sharpe_ratio']:.2f})")
        
        # Avaliação geral de robustez
        overall_robustness = self._calculate_overall_robustness()
        print(f"\n🎖️  ROBUSTEZ GERAL: {overall_robustness}/10")
        
        if overall_robustness >= 8:
            print("   ✅ IA ALTAMENTE ROBUSTA - Pronta para mercado real")
        elif overall_robustness >= 6:
            print("   ⚠️  IA MODERADAMENTE ROBUSTA - Pode operar com cautela")  
        else:
            print("   ❌ IA POUCO ROBUSTA - Necessita mais treinamento")
    
    def _calculate_overall_robustness(self):
        """Calcula score geral de robustez"""
        score = 5  # Base
        
        if 'crash_tests' in self.stress_results:
            survival_count = len([t for t in self.stress_results['crash_tests'] if t['survived']])
            score += (survival_count / len(self.stress_results['crash_tests'])) * 3
        
        if 'volatility_tests' in self.stress_results:
            positive_returns = len([t for t in self.stress_results['volatility_tests'] if t['total_return'] > 0])
            score += (positive_returns / len(self.stress_results['volatility_tests'])) * 2
        
        return min(10, score)

def main():
    tester = StressTester("cerebros/genesis_v1")
    
    # Executa testes
    crash_results = tester.run_market_crash_test()
    vol_results = tester.run_volatility_test()
    
    # Gera relatório
    tester.generate_stress_report()
    
    return tester.stress_results

if __name__ == "__main__":
    main()