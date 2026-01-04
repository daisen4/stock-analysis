#!/usr/bin/env python3
"""
Asset Return Analysis Tool
各アセット（株式、債券、金、原油）の年別・月別騰落率を分析
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 分析対象アセット
ASSETS = {
    'SPY': 'S&P500 (米国株)',
    'TLT': '米国長期国債',
    'GLD': '金',
    'USO': '原油'
}

class AssetAnalyzer:
    def __init__(self, tickers):
        self.tickers = tickers
        self.data = {}
        self.annual_returns = None
        self.monthly_returns = None

    def fetch_data(self):
        """Yahoo Finance APIから直接データ取得"""
        print("データを取得中（Yahoo Finance API）...")
        for ticker in self.tickers:
            print(f"  {ticker} ({ASSETS[ticker]})...")
            try:
                # Yahoo Finance APIを直接使用（過去20年分の日次データ）
                # 注: range=maxだと月次データになってしまうため、20yを指定
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=20y"
                headers = {'User-Agent': 'Mozilla/5.0'}

                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                        result = data['chart']['result'][0]
                        timestamps = result['timestamp']
                        closes = result['indicators']['quote'][0]['close']

                        # DataFrameに変換
                        dates = pd.to_datetime(timestamps, unit='s')
                        prices = pd.Series(closes, index=dates)

                        # NaNを除去
                        prices = prices.dropna()

                        if len(prices) > 0:
                            self.data[ticker] = prices
                            print(f"    期間: {prices.index[0].date()} ～ {prices.index[-1].date()} ({len(prices)}日)")
                        else:
                            print(f"    データが空です")
                    else:
                        print(f"    APIレスポンスが不正です")
                else:
                    print(f"    HTTPエラー: {response.status_code}")
            except Exception as e:
                print(f"    エラー: {e}")
        print()

    def calculate_annual_returns(self):
        """年別リターンを計算"""
        print("年別リターンを計算中...")
        annual_returns = {}

        for ticker, prices in self.data.items():
            yearly_ret = {}

            # 年ごとにグループ化
            for year in prices.index.year.unique():
                year_data = prices[prices.index.year == year]
                if len(year_data) > 1:
                    # 年初と年末の価格でリターン計算
                    start_price = year_data.iloc[0]
                    end_price = year_data.iloc[-1]
                    ret = (end_price - start_price) / start_price * 100
                    yearly_ret[year] = ret

            annual_returns[ticker] = yearly_ret

        # DataFrameに変換
        self.annual_returns = pd.DataFrame(annual_returns)
        print(f"  {len(self.annual_returns)}年分のデータを計算しました\n")

    def calculate_monthly_returns(self):
        """月別リターンを計算"""
        print("月別リターンを計算中...")
        monthly_returns = {}

        for ticker, prices in self.data.items():
            monthly_ret = {}

            # 月ごとにグループ化
            for year_month in prices.resample('M').last().index:
                month_data = prices[prices.index.to_period('M') == year_month.to_period('M')]
                if len(month_data) > 1:
                    start_price = month_data.iloc[0]
                    end_price = month_data.iloc[-1]
                    ret = (end_price - start_price) / start_price * 100
                    monthly_ret[year_month] = ret

            monthly_returns[ticker] = monthly_ret

        # DataFrameに変換
        self.monthly_returns = pd.DataFrame(monthly_returns)
        print(f"  {len(self.monthly_returns)}ヶ月分のデータを計算しました\n")

    def print_statistics(self):
        """統計情報を表示"""
        print("="*80)
        print("統計サマリー")
        print("="*80)

        for ticker in self.annual_returns.columns:
            returns = self.annual_returns[ticker].dropna()
            print(f"\n{ticker} ({ASSETS[ticker]})")
            print(f"  データ期間: {len(returns)}年")
            print(f"  平均年次リターン: {returns.mean():.2f}%")
            print(f"  標準偏差: {returns.std():.2f}%")
            print(f"  最高リターン: {returns.max():.2f}% ({returns.idxmax()}年)")
            print(f"  最低リターン: {returns.min():.2f}% ({returns.idxmin()}年)")
            print(f"  プラスの年: {(returns > 0).sum()}年 ({(returns > 0).sum() / len(returns) * 100:.1f}%)")

        print("\n" + "="*80 + "\n")

    def create_annual_returns_table(self):
        """年別リターンのテーブルを作成"""
        print("年別リターン表を作成中...")

        fig, ax = plt.subplots(figsize=(12, max(8, len(self.annual_returns) * 0.3)))

        # テーブルデータの準備
        table_data = self.annual_returns.copy()
        table_data = table_data.round(2)

        # セルの色を設定（プラスは緑、マイナスは赤）
        colors = []
        for _, row in table_data.iterrows():
            row_colors = []
            for val in row:
                if pd.isna(val):
                    row_colors.append('white')
                elif val > 0:
                    intensity = min(abs(val) / 50, 1)
                    row_colors.append((0.8, 1 - intensity * 0.4, 0.8))  # 緑系
                else:
                    intensity = min(abs(val) / 50, 1)
                    row_colors.append((1, 0.8 - intensity * 0.4, 0.8))  # 赤系
            colors.append(row_colors)

        # テーブル作成
        table = ax.table(cellText=table_data.values,
                        rowLabels=table_data.index,
                        colLabels=[f"{t}\n{ASSETS[t]}" for t in table_data.columns],
                        cellColours=colors,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        ax.axis('off')
        ax.set_title('年別リターン（%）- Yahoo Finance実データ', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('/Users/daisen4/Project/stock_analysis/asset_returns_table.png', dpi=150, bbox_inches='tight')
        print("  保存: asset_returns_table.png\n")

    def create_heatmap(self):
        """年別リターンのヒートマップを作成"""
        print("ヒートマップを作成中...")

        fig, ax = plt.subplots(figsize=(10, max(8, len(self.annual_returns) * 0.4)))

        # ヒートマップ作成
        sns.heatmap(self.annual_returns, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=0, cbar_kws={'label': 'リターン (%)'}, ax=ax,
                   linewidths=0.5, linecolor='gray')

        ax.set_xlabel('')
        ax.set_ylabel('年', fontsize=12)
        ax.set_title('年別リターン ヒートマップ - Yahoo Finance実データ', fontsize=16, fontweight='bold')

        # X軸ラベルをアセット名に
        ax.set_xticklabels([f"{t}\n{ASSETS[t]}" for t in self.annual_returns.columns], rotation=0)

        plt.tight_layout()
        plt.savefig('/Users/daisen4/Project/stock_analysis/asset_heatmap.png', dpi=150, bbox_inches='tight')
        print("  保存: asset_heatmap.png\n")

    def create_cumulative_returns_plot(self):
        """累積リターンの折れ線グラフを作成"""
        print("累積リターンのグラフを作成中...")

        fig, ax = plt.subplots(figsize=(14, 8))

        # 各アセットの累積リターンを計算
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, ticker in enumerate(self.data.keys()):
            prices = self.data[ticker]
            # 最初の価格を100として正規化
            normalized = (prices / prices.iloc[0]) * 100
            ax.plot(normalized.index, normalized.values,
                   label=f"{ticker} ({ASSETS[ticker]})",
                   linewidth=2, color=colors[i])

        ax.set_xlabel('日付', fontsize=12)
        ax.set_ylabel('累積リターン (100=開始時)', fontsize=12)
        ax.set_title('資産クラス別 累積リターンの推移', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 対数スケール

        plt.tight_layout()
        plt.savefig('/Users/daisen4/Project/stock_analysis/cumulative_returns.png', dpi=150, bbox_inches='tight')
        print("  保存: cumulative_returns.png\n")

    def create_annual_bar_chart(self):
        """年別リターンの棒グラフを作成"""
        print("年別リターンの棒グラフを作成中...")

        fig, ax = plt.subplots(figsize=(16, 8))

        # 棒グラフの設定
        x = np.arange(len(self.annual_returns.index))
        width = 0.2
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, ticker in enumerate(self.annual_returns.columns):
            values = self.annual_returns[ticker].values
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, values, width,
                         label=f"{ticker} ({ASSETS[ticker]})",
                         color=colors[i], alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('年', fontsize=12)
        ax.set_ylabel('年次リターン (%)', fontsize=12)
        ax.set_title('資産クラス別 年次リターン', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.annual_returns.index, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('/Users/daisen4/Project/stock_analysis/annual_returns_bar.png', dpi=150, bbox_inches='tight')
        print("  保存: annual_returns_bar.png\n")

    def create_monthly_heatmap(self):
        """月別リターンのヒートマップを作成"""
        print("月別リターンのヒートマップを作成中...")

        # 月別リターンを年×月の形式に変換
        for ticker in self.monthly_returns.columns:
            monthly_pivot = self.monthly_returns[ticker].reset_index()
            monthly_pivot.columns = ['date', 'return']
            monthly_pivot['year'] = monthly_pivot['date'].dt.year
            monthly_pivot['month'] = monthly_pivot['date'].dt.month

            pivot_table = monthly_pivot.pivot(index='year', columns='month', values='return')

            fig, ax = plt.subplots(figsize=(14, max(8, len(pivot_table) * 0.3)))

            sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn',
                       center=0, cbar_kws={'label': 'リターン (%)'}, ax=ax,
                       linewidths=0.5, linecolor='gray')

            ax.set_xlabel('月', fontsize=12)
            ax.set_ylabel('年', fontsize=12)
            ax.set_title(f'{ticker} ({ASSETS[ticker]}) - 月別リターン',
                        fontsize=16, fontweight='bold')
            # 実際の月数に応じてラベルを設定
            month_labels = [int(m) for m in pivot_table.columns]
            ax.set_xticklabels(month_labels)

            plt.tight_layout()
            filename = f'/Users/daisen4/Project/stock_analysis/monthly_returns_{ticker}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  保存: monthly_returns_{ticker}.png")

        print()

    def run_analysis(self):
        """分析を実行"""
        self.fetch_data()

        if not self.data:
            print("データが取得できませんでした")
            return

        self.calculate_annual_returns()
        self.calculate_monthly_returns()
        self.print_statistics()

        # 可視化
        self.create_annual_returns_table()
        self.create_heatmap()
        self.create_cumulative_returns_plot()
        self.create_annual_bar_chart()
        self.create_monthly_heatmap()

        print("="*80)
        print("分析完了！")
        print("="*80)

def main():
    tickers = list(ASSETS.keys())
    analyzer = AssetAnalyzer(tickers)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
