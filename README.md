# PSO-based Feature Selection

粒子群最適化（PSO）を用いた特徴量選択と、選択前後での回帰モデル性能（MSE）の比較を行う実験コードです。k-fold クロスバリデーションに対応しています。

## ディレクトリ構成

```text
.
├── main.py              # 実行スクリプト
├── data/                # データセット置き場（CSV）
├── config/
│   ├── param.yaml       # PSO・実験パラメータ設定
│   └── logging.yaml     # logging 設定
├── results/
│   └── YYYYMMDD_HHMMSS/ # 実験結果（自動生成）
│       ├── config.yaml
│       ├── YYYYMMDD_HHMMSS.log
│       └── convergence_curve.png
└── logs/                # バックアップログ
```

## 必要環境

- Python 3.9 以上（推奨）
- 主要ライブラリ
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - tqdm
  - pyyaml

```bash
pip install numpy pandas matplotlib scikit-learn tqdm pyyaml
```

## データ形式

`data/` 以下に CSV ファイルを配置します。

- **1列目**: サンプルID等（使用しない）
- **2列目〜最後の1列手前**: 特徴量
- **最終列**: 目的変数（回帰）

例：

```text
id, x1, x2, x3, ..., y
```

## 実行方法

```bash
python main.py
```

実行すると以下が自動的に行われます：

1. PSO による特徴量選択
2. k-fold CV による性能評価
3. ログ・設定ファイルの保存
4. 収束曲線の可視化

## PSO の概要

- **粒子表現**
  各粒子は 0/1 のバイナリベクトルで特徴量の選択を表現
- **評価関数**
  - MSE
  - 特徴量数に対するペナルティ付き
    ```text
    MSE × (1 + α × (#features / dim))
    ```
- **トポロジー**
  - global
  - ring
  - von_neumann
  - random
- **ランダムリブート**
  - 一定反復ごとに停滞粒子を再初期化

## 出力結果

- **ログ**
  - 各 fold の最良特徴量
  - PSO 前後の MSE
- **可視化**
  - `convergence_curve.png`
    各 fold における global best MSE の推移

## パラメータ設定（param.yaml）

主な設定項目：

- PSO
  - `num_particles`
  - `num_iterations`
  - `w`, `c1`, `c2`
  - `topology_type`
- 評価
  - `eval_type`（例: mse）
  - `eval_model_type`（lr / svr）
- CV
  - `k_fold`
  - `shuffle`
  - `random_seed`

## 注意点

- 特徴量が 0 個の場合は大きなペナルティを与えています
- PSO の性質上、実行ごとに結果が多少変動します
- 大規模次元では計算コストに注意してください

## 今後の拡張案

- 分類タスク対応
- マルチ目的 PSO（精度 × 次元削減）
- 並列化による高速化