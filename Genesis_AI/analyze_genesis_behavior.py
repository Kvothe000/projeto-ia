# Genesis_AI/analyze_genesis_behavior.py
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from collections import Counter

class BehaviorAnalyzer:
    def __init__(self, model_path, feature_names):
        self.model = PPO.load(model_path)
        self.feature_names = feature_names
        self.analysis = {}
    
    def analyze_action_patterns(self, test_samples=1000):
        """Analisa padrões de ação da IA"""
        print("🔍 ANALISANDO PADRÕES DE COMPORTAMENTO...")
        
        # Gera observações aleatórias para testar comportamento
        random_observations = np.random.normal(0, 1, (test_samples, len(self.feature_names)))
        
        actions = []
        confidences = []
        
        for obs in random_observations:
            action, _ = self.model.predict(obs, deterministic=False)
            actions.append(action)
            
            # Estima confiança (probabilidade da ação)
            action_probs = self.model.policy.get_distribution(obs).distribution.probs
            confidence = action_probs.max().item()
            confidences.append(confidence)
        
        # Analisa distribuição de ações
        action_counts = Counter(actions)
        total_actions = sum(action_counts.values())
        
        print("\n📊 DISTRIBUIÇÃO DE AÇÕES:")
        for action, count in sorted(action_counts.items()):
            percentage = (count / total_actions) * 100
            action_name = self._get_action_name(action)
            print(f"   {action_name}: {count} ({percentage:.1f}%)")
        
        # Análise de confiança
        avg_confidence = np.mean(confidences)
        high_confidence_ratio = len([c for c in confidences if c > 0.7]) / len(confidences)
        
        print(f"\n🎯 ANÁLISE DE CONFIANÇA:")
        print(f"   Confiança Média: {avg_confidence:.3f}")
        print(f"   Ações com Alta Confiança (>70%): {high_confidence_ratio:.1%}")
        
        self.analysis.update({
            'action_distribution': action_counts,
            'avg_confidence': avg_confidence,
            'high_confidence_ratio': high_confidence_ratio,
            'action_names': [self._get_action_name(a) for a in actions[:100]]  # Amostra
        })
        
        self._plot_behavior_analysis(actions, confidences)
        
        return self.analysis
    
    def _get_action_name(self, action):
        """Traduz código de ação para nome"""
        action_names = {
            0: "HOLD",
            1: "LONG 25%", 
            2: "LONG 50%",
            3: "CLOSE"
        }
        return action_names.get(action, f"Ação {action}")
    
    def _plot_behavior_analysis(self, actions, confidences):
        """Plota análise de comportamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico de distribuição de ações
        action_counts = Counter(actions)
        action_names = [self._get_action_name(a) for a in action_counts.keys()]
        ax1.bar(action_names, action_counts.values())
        ax1.set_title('Distribuição de Ações da IA')
        ax1.set_ylabel('Frequência')
        ax1.tick_params(axis='x', rotation=45)
        
        # Gráfico de distribuição de confiança
        ax2.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribuição de Confiança da IA')
        ax2.set_xlabel('Nível de Confiança')
        ax2.set_ylabel('Frequência')
        ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Média: {np.mean(confidences):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('Genesis_AI/behavior_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Gráficos de comportamento salvos em: Genesis_AI/behavior_analysis.png")
    
    def analyze_decision_making(self, market_scenarios):
        """Analisa como a IA toma decisões em diferentes cenários"""
        print("\n🎲 ANALISANDO TOMADA DE DECISÃO...")
        
        scenario_results = []
        
        for scenario_name, scenario_data in market_scenarios.items():
            actions = []
            for obs in scenario_data:
                action, _ = self.model.predict(obs, deterministic=True)
                actions.append(action)
            
            # Análise por cenário
            action_dist = Counter(actions)
            most_common_action = action_dist.most_common(1)[0][0]
            
            scenario_results.append({
                'scenario': scenario_name,
                'most_common_action': self._get_action_name(most_common_action),
                'action_distribution': action_dist,
                'aggressiveness': self._calculate_aggressiveness(actions)
            })
        
        # Exibe resultados
        for result in scenario_results:
            print(f"\n📈 Cenário: {result['scenario']}")
            print(f"   Ação Mais Comum: {result['most_common_action']}")
            print(f"   Agressividade: {result['aggressiveness']:.3f}")
            
        return scenario_results
    
    def _calculate_aggressiveness(self, actions):
        """Calcula índice de agressividade baseado nas ações"""
        aggressive_actions = [1, 2]  # LONG positions
        return len([a for a in actions if a in aggressive_actions]) / len(actions)

def create_test_scenarios(feature_count):
    """Cria cenários de mercado para teste"""
    scenarios = {
        'Mercado em Alta': np.random.normal(1, 0.1, (50, feature_count)),
        'Mercado em Baixa': np.random.normal(-1, 0.1, (50, feature_count)),
        'Mercado Lateral': np.random.normal(0, 0.05, (50, feature_count)),
        'Alta Volatilidade': np.random.normal(0, 2, (50, feature_count))
    }
    return scenarios

def main():
    # Nomes das features (ajuste conforme seu dataset)
    feature_names = [
        'mom_3', 'mom_5', 'mom_10', 'vol_ratio', 'pos_canal',
        'trend_str', 'vol_surge', 'btc_mom', 'rel_str'
    ]
    
    analyzer = BehaviorAnalyzer(
        model_path="cerebros/genesis_v2_stable",
        feature_names=feature_names
    )
    
    # Análise de padrões gerais
    behavior_analysis = analyzer.analyze_action_patterns()
    
    # Análise por cenários
    scenarios = create_test_scenarios(len(feature_names))
    decision_analysis = analyzer.analyze_decision_making(scenarios)
    
    print("\n✅ Análise de comportamento concluída!")
    return behavior_analysis, decision_analysis

if __name__ == "__main__":
    main()