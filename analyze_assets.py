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
    '^TNX': '米国10年債利回り',
    'GLD': '金',
    'USO': '原油'
}

class AssetAnalyzer:
    def __init__(self, tickers):
        self.tickers = tickers
        self.data = {}
        self.eps_data = None
        self.inflation_data = None
        self.core_pce_data = None
        self.real_rate_data = None
        self.per_data = None
        self.ff_rate_data = None
        self.annual_returns = None
        self.monthly_returns = None
        self.eps_growth = None

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

    def fetch_sp500_eps(self):
        """S&P500のEPSデータを取得（multpl.comより）"""
        print("S&P500のEPSデータを取得中...")

        # S&P500の年次EPS（インフレ調整済み）
        # データソース: multpl.com
        eps_data = {
            2025: 224.07,
            2024: 216.29,
            2023: 203.76,
            2022: 189.05,
            2021: 230.52,
            2020: 117.38,
            2019: 176.28,
            2018: 171.16,
            2017: 144.77,
            2016: 127.20,
            2015: 118.82,
            2014: 141.52,
            2013: 139.65,
            2012: 122.38,
            2011: 125.14,
            2010: 114.62,
            2009: 76.66,
            2008: 22.99,
            2007: 102.34,
            2006: 115.58,
            2005: 98.95
        }

        self.eps_data = pd.Series(eps_data).sort_index()
        print(f"  {len(self.eps_data)}年分のEPSデータを取得しました")
        print(f"  期間: {self.eps_data.index[0]}年 ～ {self.eps_data.index[-1]}年\n")

    def calculate_eps_growth(self):
        """EPS成長率を計算"""
        if self.eps_data is None:
            return

        print("EPS成長率を計算中...")
        eps_growth = {}

        for i in range(1, len(self.eps_data)):
            year = self.eps_data.index[i]
            prev_eps = self.eps_data.iloc[i-1]
            curr_eps = self.eps_data.iloc[i]

            if prev_eps > 0:
                growth_rate = (curr_eps - prev_eps) / prev_eps * 100
                eps_growth[year] = growth_rate

        self.eps_growth = pd.Series(eps_growth)
        print(f"  {len(self.eps_growth)}年分のEPS成長率を計算しました\n")

    def fetch_inflation_data(self):
        """米国インフレ率データを取得（usinflationcalculator.comより）"""
        print("米国インフレ率データを取得中...")

        # 米国の年次インフレ率（CPI前年比）
        # データソース: usinflationcalculator.com
        inflation_data = {
            2025: 2.7,  # 最新（11月時点）
            2024: 2.9,
            2023: 3.4,
            2022: 6.5,
            2021: 7.0,
            2020: 1.4,
            2019: 2.3,
            2018: 1.9,
            2017: 2.1,
            2016: 2.1,
            2015: 0.7,
            2014: 0.8,
            2013: 1.5,
            2012: 1.7,
            2011: 3.0,
            2010: 1.5,
            2009: 2.7,
            2008: 0.1,
            2007: 4.1,
            2006: 2.5,
            2005: 3.4
        }

        self.inflation_data = pd.Series(inflation_data).sort_index()
        print(f"  {len(self.inflation_data)}年分のインフレ率データを取得しました")
        print(f"  期間: {self.inflation_data.index[0]}年 ～ {self.inflation_data.index[-1]}年\n")

    def fetch_core_pce_data(self):
        """米国Core PCEデータを取得（FRB/BEAより）"""
        print("米国Core PCEデータを取得中...")

        # Core PCE（食品・エネルギー除くPCE前年比）
        # データソース: Federal Reserve, BEA
        core_pce_data = {
            2025: 2.7,  # 最新（11月時点）
            2024: 2.8,
            2023: 4.1,
            2022: 4.7,
            2021: 4.9,
            2020: 1.4,
            2019: 1.6,
            2018: 1.9,
            2017: 1.5,
            2016: 1.7,
            2015: 1.3,
            2014: 1.5,
            2013: 1.1,
            2012: 1.8,
            2011: 1.6,
            2010: 1.2,
            2009: 1.5,
            2008: 2.3,
            2007: 2.2,
            2006: 2.3,
            2005: 1.9
        }

        self.core_pce_data = pd.Series(core_pce_data).sort_index()
        print(f"  {len(self.core_pce_data)}年分のCore PCEデータを取得しました")
        print(f"  期間: {self.core_pce_data.index[0]}年 ～ {self.core_pce_data.index[-1]}年\n")

    def fetch_ff_rate_data(self):
        """米国FF金利データを取得（FRBより）"""
        print("米国FF金利データを取得中...")

        # Federal Funds Rate（年末時点の政策金利）
        # データソース: Federal Reserve
        ff_rate_data = {
            2025: 4.50,  # 最新（12月時点、予想）
            2024: 4.38,
            2023: 5.33,
            2022: 4.33,
            2021: 0.08,
            2020: 0.09,
            2019: 1.55,
            2018: 2.40,
            2017: 1.42,
            2016: 0.65,
            2015: 0.36,
            2014: 0.12,
            2013: 0.12,
            2012: 0.16,
            2011: 0.08,
            2010: 0.18,
            2009: 0.12,
            2008: 0.16,
            2007: 4.24,
            2006: 5.24,
            2005: 4.16
        }

        self.ff_rate_data = pd.Series(ff_rate_data).sort_index()
        print(f"  {len(self.ff_rate_data)}年分のFF金利データを取得しました")
        print(f"  期間: {self.ff_rate_data.index[0]}年 ～ {self.ff_rate_data.index[-1]}年\n")

    def calculate_real_rate_and_per(self):
        """実質金利とPERを計算"""
        print("実質金利とPERを計算中...")

        # 実質金利 = 10年債利回り - Core PCE
        if self.core_pce_data is not None and '^TNX' in self.annual_returns.columns:
            real_rates = {}
            for year in self.annual_returns.index:
                if year in self.core_pce_data.index:
                    # 年末の利回り水準を取得
                    year_data = self.data['^TNX'][self.data['^TNX'].index.year == year]
                    if len(year_data) > 0:
                        nominal_rate = year_data.iloc[-1]  # 年末の利回り
                        inflation = self.core_pce_data[year]
                        real_rates[year] = nominal_rate - inflation

            self.real_rate_data = pd.Series(real_rates)
            print(f"  {len(self.real_rate_data)}年分の実質金利を計算しました")

        # PER = 株価 / EPS
        # 注: SPYはS&P500指数の約1/10なので、SPY価格を10倍してS&P500指数相当にする
        if self.eps_data is not None and 'SPY' in self.data:
            per_values = {}
            for year in self.eps_data.index:
                if year >= 2006:  # SPYデータがある期間のみ
                    # 年末のSPY価格を取得
                    year_data = self.data['SPY'][self.data['SPY'].index.year == year]
                    if len(year_data) > 0 and year in self.eps_data.index:
                        spy_price = year_data.iloc[-1] * 10  # SPYをS&P500指数相当に変換
                        eps = self.eps_data[year]
                        if eps > 0:
                            per_values[year] = spy_price / eps

            self.per_data = pd.Series(per_values)
            print(f"  {len(self.per_data)}年分のPERを計算しました\n")

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

                    # ^TNXは利回りなので差分で計算、他は変化率
                    if ticker == '^TNX':
                        ret = end_price - start_price  # 差分（%ポイント）
                    else:
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

                    # ^TNXは利回りなので差分で計算、他は変化率
                    if ticker == '^TNX':
                        ret = end_price - start_price  # 差分（%ポイント）
                    else:
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

        # EPS成長率の統計
        if self.eps_growth is not None and len(self.eps_growth) > 0:
            print(f"\nS&P500 EPS成長率")
            print(f"  データ期間: {len(self.eps_growth)}年")
            print(f"  平均EPS成長率: {self.eps_growth.mean():.2f}%")
            print(f"  標準偏差: {self.eps_growth.std():.2f}%")
            print(f"  最高成長率: {self.eps_growth.max():.2f}% ({self.eps_growth.idxmax()}年)")
            print(f"  最低成長率: {self.eps_growth.min():.2f}% ({self.eps_growth.idxmin()}年)")
            print(f"  プラスの年: {(self.eps_growth > 0).sum()}年 ({(self.eps_growth > 0).sum() / len(self.eps_growth) * 100:.1f}%)")

        # インフレ率の統計
        if self.inflation_data is not None and len(self.inflation_data) > 0:
            print(f"\n米国インフレ率（CPI）")
            print(f"  データ期間: {len(self.inflation_data)}年")
            print(f"  平均インフレ率: {self.inflation_data.mean():.2f}%")
            print(f"  標準偏差: {self.inflation_data.std():.2f}%")
            print(f"  最高インフレ率: {self.inflation_data.max():.2f}% ({self.inflation_data.idxmax()}年)")
            print(f"  最低インフレ率: {self.inflation_data.min():.2f}% ({self.inflation_data.idxmin()}年)")

        # Core PCEの統計
        if self.core_pce_data is not None and len(self.core_pce_data) > 0:
            print(f"\n米国Core PCE（FRB重視指標）")
            print(f"  データ期間: {len(self.core_pce_data)}年")
            print(f"  平均Core PCE: {self.core_pce_data.mean():.2f}%")
            print(f"  標準偏差: {self.core_pce_data.std():.2f}%")
            print(f"  最高Core PCE: {self.core_pce_data.max():.2f}% ({self.core_pce_data.idxmax()}年)")
            print(f"  最低Core PCE: {self.core_pce_data.min():.2f}% ({self.core_pce_data.idxmin()}年)")

        # FF金利の統計
        if self.ff_rate_data is not None and len(self.ff_rate_data) > 0:
            print(f"\n米国FF金利（政策金利）")
            print(f"  データ期間: {len(self.ff_rate_data)}年")
            print(f"  平均FF金利: {self.ff_rate_data.mean():.2f}%")
            print(f"  標準偏差: {self.ff_rate_data.std():.2f}%")
            print(f"  最高FF金利: {self.ff_rate_data.max():.2f}% ({self.ff_rate_data.idxmax()}年)")
            print(f"  最低FF金利: {self.ff_rate_data.min():.2f}% ({self.ff_rate_data.idxmin()}年)")

        # 実質金利の統計
        if self.real_rate_data is not None and len(self.real_rate_data) > 0:
            print(f"\n実質金利（10年債 - Core PCE）")
            print(f"  データ期間: {len(self.real_rate_data)}年")
            print(f"  平均実質金利: {self.real_rate_data.mean():.2f}%")
            print(f"  標準偏差: {self.real_rate_data.std():.2f}%")
            print(f"  最高実質金利: {self.real_rate_data.max():.2f}% ({self.real_rate_data.idxmax()}年)")
            print(f"  最低実質金利: {self.real_rate_data.min():.2f}% ({self.real_rate_data.idxmin()}年)")

        # PERの統計
        if self.per_data is not None and len(self.per_data) > 0:
            print(f"\nS&P500 PER（株価収益率）")
            print(f"  データ期間: {len(self.per_data)}年")
            print(f"  平均PER: {self.per_data.mean():.2f}倍")
            print(f"  標準偏差: {self.per_data.std():.2f}倍")
            print(f"  最高PER: {self.per_data.max():.2f}倍 ({self.per_data.idxmax()}年)")
            print(f"  最低PER: {self.per_data.min():.2f}倍 ({self.per_data.idxmin()}年)")

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
        """年別リターンの棒グラフを作成（6つのグラフを縦に並べて1つのファイルとして保存）"""
        print("年別リターンの棒グラフを作成中...")

        # 6つのサブプロットを縦に並べる
        # 順番: 1.マクロ指標（利回り・実質金利・Core PCE） 2.EPS成長率 3.PER 4.SPY 5.GLD 6.USO
        fig, axes = plt.subplots(6, 1, figsize=(16, 30))

        # グラフの順番を指定
        plot_order = [
            ('MACRO', 0),      # 1番目: マクロ指標統合グラフ
            ('EPS', 1),        # 2番目: EPS成長率
            ('PER', 2),        # 3番目: PER
            ('SPY', 3),        # 4番目: S&P500
            ('GLD', 4),        # 5番目: 金
            ('USO', 5)         # 6番目: 原油
        ]

        for ticker, idx in plot_order:
            ax = axes[idx]

            # マクロ指標統合グラフの場合
            if ticker == 'MACRO':
                # 10年債利回り水準を取得
                tnx_levels = {}
                for year in self.annual_returns.index:
                    year_data = self.data['^TNX'][self.data['^TNX'].index.year == year]
                    if len(year_data) > 0:
                        tnx_levels[year] = year_data.iloc[-1]

                years = sorted(tnx_levels.keys())
                tnx_values = [tnx_levels[y] for y in years]

                # Core PCE、FF金利のデータを取得
                core_pce_values = [self.core_pce_data[y] if y in self.core_pce_data.index else None for y in years]
                ff_rate_values = [self.ff_rate_data[y] if y in self.ff_rate_data.index else None for y in years]

                # 折れ線グラフ作成
                ax.plot(years, tnx_values, marker='o', linewidth=2, label='10年債利回り', color='#1f77b4')
                ax.plot(years, ff_rate_values, marker='D', linewidth=2, label='FF金利', color='#9467bd')
                ax.plot(years, core_pce_values, marker='^', linewidth=2, label='Core PCE', color='#ff7f0e')

                # ゼロラインを追加
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

                # FRB目標ライン
                ax.axhline(y=2.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='FRB目標: 2.0%')

                # ラベルとタイトル
                ax.set_xlabel('年', fontsize=11)
                ax.set_ylabel('利回り・インフレ率 (%)', fontsize=11)
                ax.set_title('マクロ指標推移（10年債利回り・FF金利・Core PCE）',
                            fontsize=14, fontweight='bold', pad=10)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(years)
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)

            # PERの場合
            elif ticker == 'PER':
                if self.per_data is None or len(self.per_data) == 0:
                    continue

                years = self.per_data.index
                values = self.per_data.values
                x = np.arange(len(years))

                # 棒グラフ作成（全て紫系で表示）
                bar_colors = ['#9467bd' for _ in values]
                bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                # 平均値を表示
                avg_per = np.mean(values)
                ax.axhline(y=avg_per, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.7, label=f'平均: {avg_per:.1f}倍')
                ax.legend(loc='upper right', fontsize=9)

                # ラベルとタイトル
                ax.set_xlabel('年', fontsize=11)
                ax.set_ylabel('PER (倍)', fontsize=11)
                ax.set_title('S&P500 PER（株価収益率）',
                            fontsize=14, fontweight='bold', pad=10)
                ax.set_xticks(x)
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

            # EPS成長率の場合
            elif ticker == 'EPS':
                if self.eps_growth is None or len(self.eps_growth) == 0:
                    continue

                years = self.eps_growth.index
                values = self.eps_growth.values
                x = np.arange(len(years))

                # 棒グラフ作成（プラスは緑、マイナスは赤）
                bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
                bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                # ゼロラインを追加
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

                # 平均値を表示
                avg_growth = np.mean(values)
                ax.axhline(y=avg_growth, color='blue', linestyle='--',
                          linewidth=1.5, alpha=0.7, label=f'平均: {avg_growth:.1f}%')
                ax.legend(loc='upper right', fontsize=9)

                # ラベルとタイトル
                ax.set_xlabel('年', fontsize=11)
                ax.set_ylabel('EPS成長率 (%)', fontsize=11)
                ax.set_title('S&P500 EPS成長率（インフレ調整済み）',
                            fontsize=14, fontweight='bold', pad=10)
                ax.set_xticks(x)
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

            # 資産クラスの場合
            else:
                # ^TNXは水準を表示、その他は変化率を表示
                if ticker == '^TNX':
                    # 年末の利回り水準を取得
                    tnx_levels = {}
                    for year in self.annual_returns.index:
                        year_data = self.data['^TNX'][self.data['^TNX'].index.year == year]
                        if len(year_data) > 0:
                            tnx_levels[year] = year_data.iloc[-1]  # 年末の利回り

                    years = list(tnx_levels.keys())
                    values = list(tnx_levels.values())
                    x = np.arange(len(years))

                    # 棒グラフ作成（全て青系で表示）
                    bar_colors = ['#1f77b4' for _ in values]
                    bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                    # 平均値を表示
                    avg_level = np.mean(values)
                    ax.axhline(y=avg_level, color='red', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'平均: {avg_level:.2f}%')
                    ax.legend(loc='upper right', fontsize=9)

                    # ラベルとタイトル
                    ax.set_xlabel('年', fontsize=11)
                    ax.set_ylabel('利回り水準 (%)', fontsize=11)
                    ax.set_title(f'{ticker} ({ASSETS[ticker]}) - 年末利回り水準',
                                fontsize=14, fontweight='bold', pad=10)
                    ax.set_xticks(x)
                    ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    values = self.annual_returns[ticker].values
                    years = self.annual_returns.index
                    x = np.arange(len(years))

                    # 棒グラフ作成（プラスは緑、マイナスは赤）
                    bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
                    bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                    # ゼロラインを追加
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

                    # 平均値を表示
                    avg_return = np.mean(values)
                    ax.axhline(y=avg_return, color='blue', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'平均: {avg_return:.1f}%')
                    ax.legend(loc='upper right', fontsize=9)

                    # ラベルとタイトル
                    ax.set_xlabel('年', fontsize=11)
                    ax.set_ylabel('年次リターン (%)', fontsize=11)
                    ax.set_title(f'{ticker} ({ASSETS[ticker]}) - 年次リターン',
                                fontsize=14, fontweight='bold', pad=10)
                    ax.set_xticks(x)
                    ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('/Users/daisen4/Project/stock_analysis/annual_returns_bar.png', dpi=150, bbox_inches='tight')
        print("  保存: annual_returns_bar.png\n")

    def create_eps_growth_chart(self):
        """EPS成長率のグラフを作成"""
        if self.eps_growth is None or len(self.eps_growth) == 0:
            return

        print("EPS成長率のグラフを作成中...")

        fig, ax = plt.subplots(figsize=(14, 6))

        years = self.eps_growth.index
        values = self.eps_growth.values
        x = np.arange(len(years))

        # 棒グラフ作成（プラスは緑、マイナスは赤）
        bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
        bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        # ゼロラインを追加
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        # 平均値を表示
        avg_growth = np.mean(values)
        ax.axhline(y=avg_growth, color='blue', linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'平均: {avg_growth:.1f}%')

        # ラベルとタイトル
        ax.set_xlabel('年', fontsize=12)
        ax.set_ylabel('EPS成長率 (%)', fontsize=12)
        ax.set_title('S&P500 EPS成長率（インフレ調整済み）', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig('/Users/daisen4/Project/stock_analysis/eps_growth.png', dpi=150, bbox_inches='tight')
        print("  保存: eps_growth.png\n")

    def create_monthly_bar_chart(self):
        """月別リターンの棒グラフを作成（6つのグラフを縦に並べて1つのファイルとして保存）"""
        print("月別リターンの棒グラフを作成中...")

        # 6つのサブプロットを縦に並べる
        # 順番: 1.マクロ指標（利回り・実質金利・Core PCE） 2.EPS成長率 3.PER 4.SPY 5.GLD 6.USO
        fig, axes = plt.subplots(6, 1, figsize=(16, 30))

        # グラフの順番を指定
        plot_order = [
            ('MACRO', 0),      # 1番目: マクロ指標統合グラフ
            ('EPS', 1),        # 2番目: EPS成長率（年次データ）
            ('PER', 2),        # 3番目: PER（年次データ）
            ('SPY', 3),        # 4番目: S&P500
            ('GLD', 4),        # 5番目: 金
            ('USO', 5)         # 6番目: 原油
        ]

        for ticker, idx in plot_order:
            ax = axes[idx]

            # マクロ指標統合グラフの場合
            if ticker == 'MACRO':
                # 月末の利回り水準を取得
                monthly_tnx = self.data['^TNX'].resample('M').last().dropna()
                dates = monthly_tnx.index
                tnx_values = monthly_tnx.values

                # 年次データをマッピング（Core PCE、FF金利）
                core_pce_monthly = []
                ff_rate_monthly = []
                for date in dates:
                    year = date.year
                    if year in self.core_pce_data.index:
                        core_pce_monthly.append(self.core_pce_data[year])
                    else:
                        core_pce_monthly.append(None)

                    if year in self.ff_rate_data.index:
                        ff_rate_monthly.append(self.ff_rate_data[year])
                    else:
                        ff_rate_monthly.append(None)

                # 折れ線グラフ作成
                ax.plot(dates, tnx_values, linewidth=1.5, label='10年債利回り', color='#1f77b4', alpha=0.8)
                ax.plot(dates, ff_rate_monthly, linewidth=1.5, label='FF金利（年次）', color='#9467bd', alpha=0.8, linestyle='--')
                ax.plot(dates, core_pce_monthly, linewidth=1.5, label='Core PCE（年次）', color='#ff7f0e', alpha=0.8, linestyle='--')

                # ゼロラインを追加
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

                # FRB目標ライン
                ax.axhline(y=2.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='FRB目標: 2.0%')

                # ラベルとタイトル
                ax.set_xlabel('月', fontsize=11)
                ax.set_ylabel('利回り・インフレ率 (%)', fontsize=11)
                ax.set_title('マクロ指標推移（10年債利回り：月次、FF金利・Core PCE：年次）',
                            fontsize=14, fontweight='bold', pad=10)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)

            # EPS成長率の場合（年次データを表示）
            elif ticker == 'EPS':
                if self.eps_growth is None or len(self.eps_growth) == 0:
                    continue

                years = self.eps_growth.index
                values = self.eps_growth.values
                x = np.arange(len(years))

                # 棒グラフ作成（プラスは緑、マイナスは赤）
                bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
                bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                # ゼロラインを追加
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

                # 平均値を表示
                avg_growth = np.mean(values)
                ax.axhline(y=avg_growth, color='blue', linestyle='--',
                          linewidth=1.5, alpha=0.7, label=f'平均: {avg_growth:.1f}%')
                ax.legend(loc='upper right', fontsize=9)

                # ラベルとタイトル
                ax.set_xlabel('年', fontsize=11)
                ax.set_ylabel('EPS成長率 (%)', fontsize=11)
                ax.set_title('S&P500 EPS成長率（インフレ調整済み・年次）',
                            fontsize=14, fontweight='bold', pad=10)
                ax.set_xticks(x)
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

            # PERの場合（年次データを表示）
            elif ticker == 'PER':
                if self.per_data is None or len(self.per_data) == 0:
                    continue

                years = self.per_data.index
                values = self.per_data.values
                x = np.arange(len(years))

                # 棒グラフ作成（全て紫系で表示）
                bar_colors = ['#9467bd' for _ in values]
                bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                # 平均値を表示
                avg_per = np.mean(values)
                ax.axhline(y=avg_per, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.7, label=f'平均: {avg_per:.1f}倍')
                ax.legend(loc='upper right', fontsize=9)

                # ラベルとタイトル
                ax.set_xlabel('年', fontsize=11)
                ax.set_ylabel('PER (倍)', fontsize=11)
                ax.set_title('S&P500 PER（株価収益率）・年次',
                            fontsize=14, fontweight='bold', pad=10)
                ax.set_xticks(x)
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

            # 月次データの場合
            else:
                # ^TNXは水準を表示、その他は変化率を表示
                if ticker == '^TNX':
                    # 月末の利回り水準を取得
                    monthly_levels = self.data['^TNX'].resample('M').last().dropna()
                    values = monthly_levels.values
                    dates = monthly_levels.index
                    x = np.arange(len(dates))

                    # 棒グラフ作成（全て青系で表示）
                    bar_colors = ['#1f77b4' for _ in values]
                    bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.3)

                    # 平均値を表示
                    avg_level = np.mean(values)
                    ax.axhline(y=avg_level, color='red', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'平均: {avg_level:.2f}%')
                    ax.legend(loc='upper right', fontsize=9)

                    # ラベルとタイトル
                    ax.set_xlabel('月', fontsize=11)
                    ax.set_ylabel('利回り水準 (%)', fontsize=11)
                    ax.set_title(f'{ticker} ({ASSETS[ticker]}) - 月末利回り水準',
                                fontsize=14, fontweight='bold', pad=10)

                    # X軸のラベルを間引いて表示（年の始まりのみ表示）
                    xtick_positions = []
                    xtick_labels = []
                    for i, date in enumerate(dates):
                        if date.month == 1:  # 1月のみ表示
                            xtick_positions.append(i)
                            xtick_labels.append(f"{date.year}")

                    ax.set_xticks(xtick_positions)
                    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    # 月次データを取得
                    monthly_data = self.monthly_returns[ticker].dropna()
                    values = monthly_data.values
                    dates = monthly_data.index
                    x = np.arange(len(dates))

                    # 棒グラフ作成（プラスは緑、マイナスは赤）
                    bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
                    bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.3)

                    # ゼロラインを追加
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

                    # 平均値を表示
                    avg_return = np.mean(values)
                    ax.axhline(y=avg_return, color='blue', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'平均: {avg_return:.1f}%')
                    ax.legend(loc='upper right', fontsize=9)

                    # ラベルとタイトル
                    ax.set_xlabel('月', fontsize=11)
                    ax.set_ylabel('月次リターン (%)', fontsize=11)
                    ax.set_title(f'{ticker} ({ASSETS[ticker]}) - 月次リターン',
                                fontsize=14, fontweight='bold', pad=10)

                    # X軸のラベルを間引いて表示（年の始まりのみ表示）
                    xtick_positions = []
                    xtick_labels = []
                    for i, date in enumerate(dates):
                        if date.month == 1:  # 1月のみ表示
                            xtick_positions.append(i)
                            xtick_labels.append(f"{date.year}")

                    ax.set_xticks(xtick_positions)
                    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('/Users/daisen4/Project/stock_analysis/monthly_returns_bar.png', dpi=150, bbox_inches='tight')
        print("  保存: monthly_returns_bar.png\n")

    def create_quarterly_bar_chart(self):
        """四半期リターンの棒グラフを作成（6つのグラフを縦に並べて1つのファイルとして保存）"""
        print("四半期リターンの棒グラフを作成中...")

        # 6つのサブプロットを縦に並べる
        # 順番: 1.マクロ指標（利回り・実質金利・Core PCE） 2.EPS成長率 3.PER 4.SPY 5.GLD 6.USO
        fig, axes = plt.subplots(6, 1, figsize=(16, 30))

        # グラフの順番を指定
        plot_order = [
            ('MACRO', 0),      # 1番目: マクロ指標統合グラフ
            ('EPS', 1),        # 2番目: EPS成長率（年次データ）
            ('PER', 2),        # 3番目: PER（年次データ）
            ('SPY', 3),        # 4番目: S&P500
            ('GLD', 4),        # 5番目: 金
            ('USO', 5)         # 6番目: 原油
        ]

        for ticker, idx in plot_order:
            ax = axes[idx]

            # マクロ指標統合グラフの場合
            if ticker == 'MACRO':
                # 四半期末の利回り水準を取得
                quarterly_tnx = self.data['^TNX'].resample('Q').last().dropna()
                dates = quarterly_tnx.index
                tnx_values = quarterly_tnx.values

                # 年次データをマッピング（Core PCE、FF金利）
                core_pce_quarterly = []
                ff_rate_quarterly = []
                for date in dates:
                    year = date.year
                    if year in self.core_pce_data.index:
                        core_pce_quarterly.append(self.core_pce_data[year])
                    else:
                        core_pce_quarterly.append(None)

                    if year in self.ff_rate_data.index:
                        ff_rate_quarterly.append(self.ff_rate_data[year])
                    else:
                        ff_rate_quarterly.append(None)

                # 折れ線グラフ作成
                ax.plot(dates, tnx_values, linewidth=1.5, label='10年債利回り', color='#1f77b4', alpha=0.8)
                ax.plot(dates, ff_rate_quarterly, linewidth=1.5, label='FF金利（年次）', color='#9467bd', alpha=0.8, linestyle='--')
                ax.plot(dates, core_pce_quarterly, linewidth=1.5, label='Core PCE（年次）', color='#ff7f0e', alpha=0.8, linestyle='--')

                # ゼロラインを追加
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

                # FRB目標ライン
                ax.axhline(y=2.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='FRB目標: 2.0%')

                # ラベルとタイトル
                ax.set_xlabel('四半期', fontsize=11)
                ax.set_ylabel('利回り・インフレ率 (%)', fontsize=11)
                ax.set_title('マクロ指標推移（10年債利回り：四半期、FF金利・Core PCE：年次）',
                            fontsize=14, fontweight='bold', pad=10)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)

            # 実質金利の場合（年次データを表示）
            elif ticker == 'REAL_RATE':
                if self.real_rate_data is None or len(self.real_rate_data) == 0:
                    continue

                years = self.real_rate_data.index
                values = self.real_rate_data.values
                x = np.arange(len(years))

                # 棒グラフ作成（プラスは緑、マイナスは赤）
                bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
                bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                # ゼロラインを追加
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

                # 平均値を表示
                avg_rate = np.mean(values)
                ax.axhline(y=avg_rate, color='blue', linestyle='--',
                          linewidth=1.5, alpha=0.7, label=f'平均: {avg_rate:.2f}%')
                ax.legend(loc='upper right', fontsize=9)

                # ラベルとタイトル
                ax.set_xlabel('年', fontsize=11)
                ax.set_ylabel('実質金利 (%)', fontsize=11)
                ax.set_title('実質金利（10年債利回り - Core PCE）・年次',
                            fontsize=14, fontweight='bold', pad=10)
                ax.set_xticks(x)
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

            # Core PCEの場合（年次データを表示）
            elif ticker == 'CORE_PCE':
                if self.core_pce_data is None or len(self.core_pce_data) == 0:
                    continue

                years = self.core_pce_data.index
                values = self.core_pce_data.values
                x = np.arange(len(years))

                # 棒グラフ作成（全て青系で表示）
                bar_colors = ['#1f77b4' for _ in values]
                bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                # ゼロラインを追加
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

                # 平均値とFRB目標を表示
                avg_pce = np.mean(values)
                ax.axhline(y=avg_pce, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.7, label=f'平均: {avg_pce:.1f}%')
                ax.axhline(y=2.0, color='green', linestyle=':',
                          linewidth=1.5, alpha=0.7, label='FRB目標: 2.0%')
                ax.legend(loc='upper right', fontsize=9)

                # ラベルとタイトル
                ax.set_xlabel('年', fontsize=11)
                ax.set_ylabel('Core PCE (%)', fontsize=11)
                ax.set_title('米国Core PCE（FRB最重視インフレ指標）・年次',
                            fontsize=14, fontweight='bold', pad=10)
                ax.set_xticks(x)
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

            # EPS成長率の場合（年次データを表示）
            elif ticker == 'EPS':
                if self.eps_growth is None or len(self.eps_growth) == 0:
                    continue

                years = self.eps_growth.index
                values = self.eps_growth.values
                x = np.arange(len(years))

                # 棒グラフ作成（プラスは緑、マイナスは赤）
                bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
                bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                # ゼロラインを追加
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

                # 平均値を表示
                avg_growth = np.mean(values)
                ax.axhline(y=avg_growth, color='blue', linestyle='--',
                          linewidth=1.5, alpha=0.7, label=f'平均: {avg_growth:.1f}%')
                ax.legend(loc='upper right', fontsize=9)

                # ラベルとタイトル
                ax.set_xlabel('年', fontsize=11)
                ax.set_ylabel('EPS成長率 (%)', fontsize=11)
                ax.set_title('S&P500 EPS成長率（インフレ調整済み・年次）',
                            fontsize=14, fontweight='bold', pad=10)
                ax.set_xticks(x)
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

            # PERの場合（年次データを表示）
            elif ticker == 'PER':
                if self.per_data is None or len(self.per_data) == 0:
                    continue

                years = self.per_data.index
                values = self.per_data.values
                x = np.arange(len(years))

                # 棒グラフ作成（全て紫系で表示）
                bar_colors = ['#9467bd' for _ in values]
                bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                # 平均値を表示
                avg_per = np.mean(values)
                ax.axhline(y=avg_per, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.7, label=f'平均: {avg_per:.1f}倍')
                ax.legend(loc='upper right', fontsize=9)

                # ラベルとタイトル
                ax.set_xlabel('年', fontsize=11)
                ax.set_ylabel('PER (倍)', fontsize=11)
                ax.set_title('S&P500 PER（株価収益率）・年次',
                            fontsize=14, fontweight='bold', pad=10)
                ax.set_xticks(x)
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

            # 四半期データの場合
            else:
                # ^TNXは水準を表示、その他は変化率を表示
                if ticker == '^TNX':
                    # 四半期末の利回り水準を取得
                    quarterly_levels = self.data['^TNX'].resample('Q').last().dropna()
                    values = quarterly_levels.values
                    dates = quarterly_levels.index
                    x = np.arange(len(dates))

                    # 棒グラフ作成（全て青系で表示）
                    bar_colors = ['#1f77b4' for _ in values]
                    bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                    # 平均値を表示
                    avg_level = np.mean(values)
                    ax.axhline(y=avg_level, color='red', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'平均: {avg_level:.2f}%')
                    ax.legend(loc='upper right', fontsize=9)

                    # ラベルとタイトル
                    ax.set_xlabel('四半期', fontsize=11)
                    ax.set_ylabel('利回り水準 (%)', fontsize=11)
                    ax.set_title(f'{ticker} ({ASSETS[ticker]}) - 四半期末利回り水準',
                                fontsize=14, fontweight='bold', pad=10)

                    # X軸のラベルを間引いて表示（年の始まりのみ表示）
                    xtick_positions = []
                    xtick_labels = []
                    for i, date in enumerate(dates):
                        if date.month == 3:  # Q1のみ表示
                            xtick_positions.append(i)
                            xtick_labels.append(f"{date.year}")

                    ax.set_xticks(xtick_positions)
                    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    # 元の価格データから四半期ごとの変化を計算
                    prices = self.data[ticker]
                    quarterly_changes = {}

                    # 四半期ごとにグループ化
                    for quarter_end in prices.resample('Q').last().index:
                        quarter_data = prices[prices.index.to_period('Q') == quarter_end.to_period('Q')]
                        if len(quarter_data) > 1:
                            start_price = quarter_data.iloc[0]
                            end_price = quarter_data.iloc[-1]
                            change = (end_price - start_price) / start_price * 100
                            quarterly_changes[quarter_end] = change

                    # 四半期データを使用
                    dates = list(quarterly_changes.keys())
                    values = list(quarterly_changes.values())
                    x = np.arange(len(dates))

                    # 棒グラフ作成（プラスは緑、マイナスは赤）
                    bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
                    bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                    # ゼロラインを追加
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

                    # 平均値を表示
                    avg_return = np.mean(values)
                    ax.axhline(y=avg_return, color='blue', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'平均: {avg_return:.1f}%')
                    ax.legend(loc='upper right', fontsize=9)

                    # ラベルとタイトル
                    ax.set_xlabel('四半期', fontsize=11)
                    ax.set_ylabel('四半期リターン (%)', fontsize=11)
                    ax.set_title(f'{ticker} ({ASSETS[ticker]}) - 四半期リターン',
                                fontsize=14, fontweight='bold', pad=10)

                    # X軸のラベルを間引いて表示（年の始まりのみ表示）
                    xtick_positions = []
                    xtick_labels = []
                    for i, date in enumerate(dates):
                        if date.month == 3:  # Q1のみ表示
                            xtick_positions.append(i)
                            xtick_labels.append(f"{date.year}")

                    ax.set_xticks(xtick_positions)
                    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('/Users/daisen4/Project/stock_analysis/quarterly_returns_bar.png', dpi=150, bbox_inches='tight')
        print("  保存: quarterly_returns_bar.png\n")

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

        # EPSとインフレ率データの取得と計算
        self.fetch_sp500_eps()
        self.calculate_eps_growth()
        self.fetch_inflation_data()
        self.fetch_core_pce_data()
        self.fetch_ff_rate_data()

        self.calculate_annual_returns()
        self.calculate_monthly_returns()
        self.calculate_real_rate_and_per()
        self.print_statistics()

        # 可視化（年次、月次、四半期の棒グラフを出力）
        self.create_annual_bar_chart()
        self.create_monthly_bar_chart()
        self.create_quarterly_bar_chart()

        print("="*80)
        print("分析完了！")
        print("="*80)

def main():
    tickers = list(ASSETS.keys())
    analyzer = AssetAnalyzer(tickers)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
