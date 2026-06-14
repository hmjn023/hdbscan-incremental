# hdbscan-incremental - アーキテクチャ

## ディレクトリ構造

```
hdbscan-incremental/
├── Cargo.toml
├── SPEC.md              # 仕様書
├── ARCHITECTURE.md      # このファイル
├── src/
│   ├── lib.rs           # 公開API
│   ├── types.rs         # 共通型定義
│   ├── distance.rs      # 距離関数
│   ├── cf.rs            # ClusteringFeature
│   ├── data_bubble.rs   # DataBubble
│   ├── bubble_tree.rs   # Bubble-tree ルート
│   ├── bubble_tree/
│   │   ├── mod.rs       # モジュール定義
│   │   ├── insert.rs    # 挿入アルゴリズム
│   │   ├── delete.rs    # 削除アルゴリズム
│   │   ├── compress.rs  # MaintainCompression (Algorithm 1)
│   │   └── split.rs     # 分割アルゴリズム (farthest pair seeding)
│   ├── hdbscan.rs       # HDBSCAN ルート
│   └── hdbscan/
│       ├── mod.rs       # モジュール定義
│       ├── core_distance.rs  # コア距離計算
│       ├── mst.rs       # MST構築 (Prim)
│       ├── dendrogram.rs    # 階層木構築
│       ├── condense.rs  # 階層の凝縮
│       ├── stability.rs # 安定性計算
│       └── eom.rs       # EOMクラスタ選択
└── tests/
    ├── integration_test.rs  # 統合テスト
    └── hdbscan_test.rs      # HDBSCANテスト
```

## モジュール依存関係

```
lib.rs
├── bubble_tree.rs
│   ├── cf.rs
│   ├── distance.rs
│   └── types.rs
├── hdbscan.rs
│   ├── data_bubble.rs
│   ├── cf.rs
│   ├── distance.rs
│   └── types.rs
└── data_bubble.rs
    ├── cf.rs
    └── distance.rs
```

## データフロー

### 1. 挿入フロー

```
add(vectors)
    │
    ├── for each vector:
    │   ├── ClusteringFeature::from_point(vector)
    │   ├── BubbleTree::insert(cf)
    │   │   ├── DFSで最適な葉を選択
    │   │   ├── 葉のCFを更新
    │   │   ├── 全祖先のCFを更新
    │   │   └── maintain_compression()
    │   │       ├── num_leaves > L → under-filled葉を削除・再挿入
    │   │       ├── num_leaves < L → over-filled葉を分割
    │   │       └── else → over-filled葉の子を再編成
    │   └── points[id] = Some(entry)
    └── return ids
```

### 2. 削除フロー

```
remove(ids)
    │
    ├── for each id:
    │   ├── entry = points[id].take()
    │   ├── BubbleTree::delete(entry.cf)
    │   │   ├── 該当ポイントを含む葉を見つける
    │   │   ├── 葉のCFからポイントを削除
    │   │   ├── 全祖先のCFを更新
    │   │   └── 葉のサイズ < m → 残りのポイントを再挿入
    │   └── maintain_compression()
    └── return
```

### 3. クラスタリングフロー

```
cluster()
    │
    ├── 1. 葉のCFを抽出 (L個)
    ├── 2. DataBubbleに変換
    │   ├── rep = LS / n
    │   ├── extent = sqrt((2*n*SS - 2*LS²) / (n*(n-1)))
    │   └── nnDist(k) = (k/n)^(1/d) * extent
    ├── 3. コア距離計算
    │   └── cd(B) = d(B, C) + C.nnDist(k)
    ├── 4. 相互到達距離計算
    │   └── d_m(B, C) = max{cd(B), cd(C), d(B, C)}
    ├── 5. MST構築 (Prim)
    ├── 6. 階層木構築 (Union-Find)
    ├── 7. 凝縮 (min_cluster_size)
    ├── 8. 安定性計算
    ├── 9. EOMクラスタ選択
    └── 10. ラベル割り当て・確率計算
```

## 主要なデータ構造

### Node (Bubble-tree)

```rust
enum Node {
    Internal {
        cf: ClusteringFeature,
        children: Vec<Box<Node>>,
        parent: Option<Weak<RefCell<Node>>>,  // 親への弱参照
    },
    Leaf {
        cf: ClusteringFeature,
        children: Vec<Box<Node>>,  // 子は実際のポイント（CFのみ）
        parent: Option<Weak<RefCell<Node>>>,
    },
}
```

### PointEntry

```rust
struct PointEntry {
    id: usize,
    vector: Vec<f64>,
    cf: ClusteringFeature,
}
```

### LinkageRow (HDBSCAN出力)

```rust
struct LinkageRow {
    left: usize,    // 左の子
    right: usize,   // 右の子
    distance: f64,  // 距離
    size: usize,    // サイズ
}
```

### CondensedTreeNode

```rust
struct CondensedTreeNode {
    parent: usize,
    child: usize,
    lambda_val: f64,
    child_size: usize,
}
```

## エラーハンドリング

```rust
#[derive(Debug, thiserror::Error)]
pub enum HdbscanError {
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },
    
    #[error("Point not found: {0}")]
    PointNotFound(usize),
    
    #[error("No points to cluster")]
    NoPoints,
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}
```

## パフォーマンス考慮

### 計算量

| 操作 | 計算量 | 備考 |
|---|---|---|
| 挿入 | O(log L * d) | L=葉数, d=次元数 |
| 削除 | O(log L * d) | 再挿入の場合あり |
| クラスタリング | O(L² * d + L² * log L) | L個のバブル間の全ペア計算 |
| maintain_compression | O(L * d) | 葉の再編成 |

### メモリ使用量

| データ構造 | メモリ使用量 |
|---|---|
| Bubble-tree | O(L * d) | L個のCF、各d次元 |
| ポイント格納 | O(N * d) | N個のベクトル |
| HDBSCAN中間結果 | O(L²) | 距離行列、linkage |

### 最適化ポイント

1. **turbovec連携**: 将来的にkNN検索をSIMD化
2. **距離計算**: コサイン距離の計算をSIMD化
3. **MST構築**: 全ペア計算を並列化 (rayon)
4. **メモリ**: ベクトルの所有権を避けて参照を使用

## テスト戦略

### 単体テスト

- `cf.rs`: CF加法性、centroid計算
- `data_bubble.rs`: extent、nnDist計算
- `distance.rs`: コサイン距離、ユークリッド距離
- `bubble_tree/insert.rs`: 挿入後のCF更新
- `bubble_tree/delete.rs`: 削除後のCF更新
- `bubble_tree/compress.rs`: 葉数維持
- `bubble_tree/split.rs`: farthest pair seeding

### 統合テスト

- 少ないポイント数でのクラスタリング
- 挿入・削除後のクラスタリング
- 既知のクラスタ構造での検証
- CCIPベクトルでの実動作

### ベンチマーク

- 挿入速度 (ポイント/秒)
- 削除速度 (ポイント/秒)
- クラスタリング速度 (バブル数依存)
- メモリ使用量
