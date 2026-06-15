# hdbscan-incremental

動的データ（挿入・削入）に対応したHDBSCANクラスタリングのRust実装。

論文 "Dynamic data summarization for hierarchical spatial clustering" (arXiv:2412.07789) に基づくBubble-treeアプローチを採用。

## 特徴

- **動的データ対応**: ベクトルの挿入・削除を効率的に処理
- **データ圧縮**: Bubble-treeによるN→L個のClustering Featureへの圧縮
- **HDBSCAN**: 密度ベースの階層クラスタリング
- **コサイン距離**: 高次元ベクトル（CCIP 768次元等）に最適

`compression_rate` は固定 leaf 数ではなく、現在の点数 `N` から `ceil(N * compression_rate)` を目標 leaf 数として計算する。

## インストール

```toml
[dependencies]
hdbscan-incremental = { git = "https://github.com/hmjn023/hdbscan-incremental" }
```

## 使い方

```rust
use hdbscan_incremental::{HdbscanIncremental, HdbscanParams};

fn main() {
    // パラメータ設定
    let params = HdbscanParams {
        min_pts: 100,
        min_cluster_size: 100,
        compression_rate: 0.01,  // 1%に圧縮
        ..Default::default()
    };

    // インデックス作成 (768次元ベクトル)
    let mut index = HdbscanIncremental::try_new(768, params).unwrap();

    // ベクトル追加
    let vectors = vec![
        vec![0.1; 768],  // キャラクターA
        vec![0.2; 768],  // キャラクターA
        vec![0.9; 768],  // キャラクターB
    ];
    let ids = index.add(&vectors).unwrap();

    // クラスタリング実行
    let result = index.cluster().unwrap();
    println!("クラスタ数: {}", result.num_clusters);
    println!("ラベル: {:?}", result.labels);

    // ベクトル削除
    index.remove(&[ids[0]]).unwrap();
}
```

## API

### `HdbscanIncremental`

| メソッド | 説明 |
|---|---|
| `try_new(dim, params)` | パラメータ検証つき新規インデックス作成 |
| `new(dim, params)` | 新規インデックス作成（不正パラメータではpanic） |
| `add(vectors)` | ベクトル追加（ID配列を返却） |
| `remove(ids)` | ベクトル削除 |
| `cluster()` | クラスタリング実行 |
| `num_bubbles()` | 現在のデータバブル数 |
| `num_points()` | 現在のポイント数 |

### `HdbscanParams`

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `min_pts` | 100 | HDBSCANの密度パラメータ |
| `min_cluster_size` | 100 | 最小クラスタサイズ |
| `compression_rate` | 0.01 | 現在の点数 N に対する目標 leaf 割合。例: 0.01 は `ceil(N * 0.01)` leaf |
| `m` | 25 | Bubble-tree最小ファンアウト |
| `cluster_selection_method` | EOM | クラスタ選択方法 (EOM/Leaf) |

### `ClusterResult`

| フィールド | 型 | 説明 |
|---|---|---|
| `labels` | `Vec<i32>` | 現在は Data Bubble 順のクラスタラベル (-1 = ノイズ)。点 ID 対応の assignment API は今後追加予定 |
| `probabilities` | `Vec<f64>` | メンバーシップ確率 |
| `num_clusters` | `usize` | クラスタ数 |
| `stability` | `Vec<f64>` | クラスタ安定性 |

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                      Online Phase                           │
│                                                             │
│  CCIPベクトル ──▶ Bubble-tree 挿入/削除                     │
│                  (データ圧縮: N → L 個のCF)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Offline Phase                           │
│                                                             │
│  L個のCF ──▶ Data Bubble変換 ──▶ HDBSCAN実行               │
│              (式3-5)              (MST→凝縮→EOM)            │
└─────────────────────────────────────────────────────────────┘
```

## テスト

```bash
cargo test
```

## 参考文献

- [Dynamic data summarization for hierarchical spatial clustering](https://arxiv.org/abs/2412.07789)
- [HDBSCAN](https://hdbscan.readthedocs.io/)
- [BIRCH](https://en.wikipedia.org/wiki/BIRCH)

## ライセンス

MIT
