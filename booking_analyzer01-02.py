import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語表示のためのライブラリ

def analyze_and_plot_peer_group_champions(csv_path, last_minute_threshold=0.5, num_tiers=3):
    """
    価格帯のピア・グループごとに「駆け込み型」戦略の成功事例を可視化する関数。

    Args:
        csv_path (str): 分析対象のCSVファイルパス
        last_minute_threshold (float): 「駆け込み型」と判断する直前予約率の閾値
        num_tiers (int): 価格帯をいくつの階層に分けるか
    """
    # --- 1. データの読み込みと前処理 ---
    print("データの読み込みと前処理を開始します...")
    try:
        df = pd.read_csv(csv_path, parse_dates=['date', 'created_at'])
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません。パスを確認してください: {csv_path}")
        return

    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = pd.to_datetime(df['date'])
    df.dropna(inplace=True)

    # --- 2. 基本指標の計算 ---
    print("基本指標（成約数、売上、最大在庫数）を計算中...")
    max_stock_df = df.groupby(['hotel_id', 'plan_id', 'room_type_id'])['stock'].max().reset_index(name='max_stock')
    df = pd.merge(df, max_stock_df, on=['hotel_id', 'plan_id', 'room_type_id'])
    df = df[df['max_stock'] >= 30].copy()

    df.sort_values(['hotel_id', 'plan_id', 'room_type_id', 'date', 'created_at'], inplace=True)
    df['stock_shift'] = df.groupby(['hotel_id', 'plan_id', 'room_type_id', 'date'])['stock'].shift(1)
    df['sold'] = (df['stock_shift'] - df['stock']).clip(lower=0).fillna(0)
    df['revenue'] = df['sold'] * df['price']

    # --- 3. ピア・グループ分析の準備 ---
    print("ピア・グループ分析の準備中...")

    # a. 宿泊日ごとのKPIを集計
    daily_kpi = df.groupby(['hotel_id', 'plan_id', 'room_type_id', 'date']).agg(
        total_revenue=('revenue', 'sum'),
        total_sold=('sold', 'sum'),
        max_stock=('max_stock', 'first')
    ).reset_index()
    daily_kpi = daily_kpi[daily_kpi['total_sold'] > 0]

    # b. RevPARとADR(平均客室単価)を計算
    daily_kpi['RevPAR'] = (daily_kpi['total_revenue'] / daily_kpi['max_stock']).fillna(0)
    daily_kpi['ADR'] = (daily_kpi['total_revenue'] / daily_kpi['total_sold']).fillna(0)

    # c. 【新規】プランごとの「代表価格(ADRの中央値)」を計算
    plan_characteristic_price = daily_kpi.groupby(['hotel_id', 'plan_id', 'room_type_id'])['ADR'].median().reset_index(name='characteristic_price')
    daily_kpi = pd.merge(daily_kpi, plan_characteristic_price, on=['hotel_id', 'plan_id', 'room_type_id'])

    # d. 【新規】代表価格を基に価格帯グループ（ピア・グループ）を作成
    # pd.qcutを使い、プランを価格帯で自動的に分類（例：3階層なら上位33%, 中位33%, 下位33%）
    try:
        tier_labels = ['松（高価格帯）', '竹（中価格帯）', '梅（低価格帯）'][::-1][:num_tiers] # 動的にラベルを生成
        daily_kpi['price_tier'] = pd.qcut(daily_kpi['characteristic_price'], q=num_tiers, labels=tier_labels, duplicates='drop')
    except ValueError:
        print("警告: プランの種類が少ないため、価格帯を細かく分類できませんでした。1つのグループとして扱います。")
        daily_kpi['price_tier'] = '単一グループ'

    # --- 4. 各ピア・グループのベストプラクティスを選定 ---
    print("各ピア・グループのベストプラクティスを選定中...")
    
    # a. 直前30日予約率を計算
    df['cutoff_date_30'] = df['date'] - pd.Timedelta(days=30)
    sold_last_30_df = df[df['created_at'] >= df['cutoff_date_30']]
    sold_last_30_agg = sold_last_30_df.groupby(['hotel_id', 'plan_id', 'room_type_id', 'date'])['sold'].sum().reset_index(name='sold_last_30')
    daily_kpi = pd.merge(daily_kpi, sold_last_30_agg, on=['hotel_id', 'plan_id', 'room_type_id', 'date'], how='left')
    daily_kpi['sold_last_30'].fillna(0, inplace=True)
    daily_kpi['last_30_days_booking_ratio'] = (daily_kpi['sold_last_30'] / daily_kpi['total_sold']).fillna(0)

    # b. 【新基準】「駆け込み型」に絞り込み
    last_minute_cases = daily_kpi[daily_kpi['last_30_days_booking_ratio'] >= last_minute_threshold]
    
    # c. 【最重要】各「価格帯グループ」の中でRevPARが最大の日を抽出
    best_dates = last_minute_cases.sort_values('RevPAR', ascending=False).groupby(['price_tier']).first().reset_index()

    # --- 5. グラフ描画 ---
    print(f"分析対象となる {len(best_dates)} 個の最適なブッキングカーブを描画します。")
    for _, g in best_dates.iterrows():
        # グラフ描画に必要な情報を元のDataFrameから取得
        sub_df = df[
            (df['hotel_id'] == g['hotel_id']) &
            (df['plan_id'] == g['plan_id']) &
            (df['room_type_id'] == g['room_type_id']) &
            (df['date'] == g['date'])
        ].copy()

        # グラフ描画用のデータ準備
        cutoff = g['date'] - pd.Timedelta(days=120)
        sold_before = sub_df[sub_df['created_at'] < cutoff]['sold'].sum()
        sub_df = sub_df[sub_df['created_at'] >= cutoff].copy()
        sub_df['created_at_norm'] = sub_df['created_at'].dt.normalize()
        
        daily_sold = sub_df.groupby('created_at_norm')['sold'].sum().sort_index().cumsum().reset_index(name='sold_cumsum')
        daily_sold['sold_cumsum'] += sold_before

        daily_price = sub_df.groupby('created_at_norm')['price'].mean().reset_index()

        # グラフ作成
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        ax1.plot(daily_sold['created_at_norm'], daily_sold['sold_cumsum'], color='mediumseagreen', label='累計予約数', marker='o', markersize=4, linestyle='-')
        ax1.set_ylabel('累計予約数', color='mediumseagreen', fontsize=12)
        ax1.axhline(y=g['max_stock'], color='grey', linestyle='--', label=f"満室ライン ({int(g['max_stock'])}室)")
        ax1.set_ylim(bottom=0)

        ax2.step(daily_price['created_at_norm'], daily_price['price'], where='post', color='tomato', label='価格')
        ax2.set_ylabel('価格 (JPY)', color='tomato', fontsize=12)
        
        title_text = (
            f"【{g['price_tier']}のベストプラクティス】 (宿泊日: {g['date'].strftime('%Y-%m-%d')})\n"
            f"Plan: {g['plan_id']}, Room: {g['room_type_id']}\n"
            f"RevPAR: {g['RevPAR']:,.0f}円 | 直前30日予約率: {g['last_30_days_booking_ratio']:.1%}"
        )
        plt.title(title_text, fontsize=14, pad=20)
        
        ax1.set_xlabel('予約日', fontsize=12)
        ax1.grid(True, linestyle=':', linewidth=0.7)
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # タイトルが重ならないように調整
        plt.show()

if __name__ == '__main__':
    # ご自身のファイルパスに変更してください
    CSV_FILE_PATH = r'D:\MyWorkspace\TempestAIProjects\.CrealProjects\hotel-data\hotel_prices.csv'
    # 価格帯を3階層に分け、「駆け込み型(直前30日予約率50%以上)」の成功事例を分析
    analyze_and_plot_peer_group_champions(CSV_FILE_PATH, last_minute_threshold=0.5, num_tiers=3)
