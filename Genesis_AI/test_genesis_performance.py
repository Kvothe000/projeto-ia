# Genesis_AI/test_genesis_performance.py
import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PerformanceTester:
    def __init__(self, model_path, test_data_path):
        self.model = PPO.load(model_path)
        self.test_data = pd.read_csv(test_data_path)
        self.results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'equity_curve': []
        }
    
    def run_backtest(self, initial_balance=10000):
        """Executa backtest completo da IA"""
        print("🧪 INICIANDO BACKTEST DA IA GENESIS...")
        
        # Prepara dados de teste (últimos 20% dos dados)
        test_size = int(0.2 * len(self.test_data))
        test_df = self.test_data.tail(test_size).copy()
        
        # Cria ambiente de teste
        env = DummyVecEnv([lambda: self._create_test_env(test_df, initial_balance)])
        
        # Executa teste
        obs = env.reset()
        done = False
        episode_rewards = []
        current_balance = initial_balance
        equity_curve = [current_balance]
        trades = []
        
        step = 0
        while not done and step < len(test_df) - 50:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step([action])
            
            episode_rewards.append(reward[0])
            current_balance = info[0].get('net_worth', current_balance)
            equity_curve.append(current_balance)
            
            # Registra trades
            if action in [1, 2]:  # Entradas LONG ou SHORT
                trades.append({
                    'step': step,
                    'action': action,
                    'reward': reward[0],
                    'balance': current_balance
                })
            
            step += 1
        
        # Analisa resultados
        self._analyze_performance(equity_curve, trades, episode_rewards, initial_balance)
        return self.results
    
    def _create_test_env(self, df, initial_balance):
        """Cria ambiente de teste especializado"""
        env = gym.make('CartPole-v1')  # Placeholder - substitua pelo seu ambiente
        # Nota: Você precisará adaptar esta parte para seu ambiente específico
        return env
    
    def _analyze_performance(self, equity_curve, trades, rewards, initial_balance):
        """Analisa métricas de performance detalhadas"""
        
        # Calcula métricas básicas
        total_return = (equity_curve[-1] - initial_balance) / initial_balance * 100
        volatility = np.std(rewards) if rewards else 0
        
        # Calcula drawdown máximo
        peak = initial_balance
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calcula Sharpe Ratio (simplificado)
        sharpe_ratio = (np.mean(rewards) / volatility * np.sqrt(252)) if volatility > 0 else 0
        
        # Conta trades vencedores
        winning_trades = len([t for t in trades if t['reward'] > 0])
        losing_trades = len([t for t in trades if t['reward'] < 0])
        
        # Preenche resultados
        self.results.update({
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / len(trades) * 100 if trades else 0,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': equity_curve,
            'avg_trade_return': np.mean([t['reward'] for t in trades]) if trades else 0
        })
    
    def generate_report(self):
        """Gera relatório completo de performance"""
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE PERFORMANCE - IA GENESIS")
        print("="*60)
        
        print(f"🎯 Total de Trades: {self.results['total_trades']}")
        print(f"✅ Trades Vencedores: {self.results['winning_trades']}")
        print(f"❌ Trades Perdedores: {self.results['losing_trades']}")
        print(f"📈 Win Rate: {self.results['win_rate']:.2f}%")
        print(f"💰 Retorno Total: {self.results['total_return']:.2f}%")
        print(f"📉 Drawdown Máximo: {self.results['max_drawdown']:.2f}%")
        print(f"⚡ Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"📊 Retorno Médio por Trade: {self.results['avg_trade_return']:.4f}")
        
        # Análise qualitativa
        win_rate = self.results['win_rate']
        total_return = self.results['total_return']
        
        if win_rate > 60 and total_return > 10:
            rating = "🌟 EXCELENTE"
        elif win_rate > 55 and total_return > 5:
            rating = "✅ BOM" 
        elif win_rate > 50 and total_return > 0:
            rating = "⚠️ REGULAR"
        else:
            rating = "❌ PRECISA DE AJUSTES"
            
        print(f"\n🎖️  CLASSIFICAÇÃO: {rating}")
        
        # Plot equity curve
        self._plot_equity_curve()
    
    def _plot_equity_curve(self):
        """Plota curva de equity"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['equity_curve'])
        plt.title('Curva de Equity - IA Genesis')
        plt.xlabel('Steps')
        plt.ylabel('Patrimônio ($)')
        plt.grid(True)
        plt.savefig('Genesis_AI/equity_curve.png')
        plt.close()
        
        print("\n📈 Gráfico salvo em: Genesis_AI/equity_curve.png")

def main():
    tester = PerformanceTester(
        model_path="cerebros/genesis_v2_stable.zip",
        test_data_path="dataset_v11_fusion.csv"
    )
    
    results = tester.run_backtest()
    tester.generate_report()
    
    return results

if __name__ == "__main__":
    main()