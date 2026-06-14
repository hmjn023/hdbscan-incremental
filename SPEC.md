# hdbscan-incremental - 仕様書

## 概要

動的データ（挿入・削除）に対応したHDBSCANクラスタリングのRust実装。
論文 "Dynamic data summarization for hierarchical spatial clustering" (arXiv:2412.07789) に基づくBubble-treeアプローチを採用。

## 目的

CCIP (Character Contrastive Learning) ベクトル（768次元）のキャラクタークラスタリングを、
動的に変化するデータセットに対して効率的に実行する。

## アーキテクチャ

### オンライン-オフラインフレームワーク

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

## データ構造

### ClusteringFeature (CF)

論文 Definition 4。BIRCH [51] 由来。

```rust
pub struct ClusteringFeature {
    pub ls: Vec<f64>,  // Linear Sum (768次元)
    pub ss: f64,        // Squared Sum (スカラー)
    pub n: usize,       // ポイント数
}
```

**加法性**: `CF_i + CF_j = {LS_i + LS_j, SS_i + SS_j, n_i + n_j}`

### DataBubble

論文 Definition 5。CFから派生。

```rust
pub struct DataBubble {
    pub rep: Vec<f64>,      // 代表点 = LS / n (式3)
    pub n: usize,           // ポイント数
    pub extent: f64,        // 拡がり (式4)
    pub nn_dist_cache: Vec<f64>,  // nnDist(k) キャッシュ (式5)
}
```

**計算式**:
- `extent = sqrt((2*n*SS - 2*LS²) / (n*(n-1)))` (式4)
- `nnDist(k) = (k/n)^(1/d) * extent` (式5)

### Bubble-tree

論文 Section 4.1。バランス木構造。

```rust
pub struct BubbleTree {
    root: Node,
    m: usize,           // 最小ファンアウト
    max_fanout: usize,  // 最大ファンアウト M (制約: 2m <= M+1)
    l: usize,           // 目標葉数 (圧縮率)
    dim: usize,         // 次元数 (768)
    total_n: usize,     // 全ポイント数
}
```

**性質**:
- Property 1: ルートは2〜M個の子を持つ
- Property 2: 内部ノードはm〜M個の子を持つ
- Property 3: 葉のCFは実際のポイントを表現、内部ノードは子のCFの集合を表現
- Property 4: 葉の数はLに維持される

## アルゴリズム

### 1. Bubble-tree 挿入

1. DFSで最適な葉を選択（repとの距離が最小）
2. 葉のCFにポイントを追加
3. 全祖先のCFを更新
4. `maintain_compression()` を呼び出し

### 2. Bubble-tree 削除

1. 該当ポイントを含む葉を見つける
2. 葉のCFからポイントを削除
3. 全祖先のCFを更新
4. 葉のサイズ < m の場合、残りのポイントを再挿入

### 3. Split (farthest pair seeding)

1. 子の中から最も遠いペア (a, b) を選択
2. 各子を a または b の近い方へ割り当て
3. 各グループが少なくとも m 個の子を持つことを保証

### 4. MaintainCompression (Algorithm 1)

```
if num_leaves > L:  // 過剰表現
    最もunder-filledな葉Uを選択
    Uを削除し、そのポイントを再挿入

elif num_leaves < L:  // 不足表現
    最もover-filledな葉Oを選択
    Oを分割して兄弟O'を作成
    O'を再挿入

else:  // 動的再編成
    最もover-filledな葉Oを選択
    Oのm個の最も遠い子を抽出して再挿入
```

### 5. Data Bubble品質指標

データサマリゼーションインデックス (式8):
```
β(B) = n / N
```

品質分類 (Chebyshev's Inequality):
- "good": β(B) ∈ [μ_β - k*σ_β, μ_β + k*σ_β]
- "under-filled": β(B) < μ_β - k*σ_β
- "over-filled": β(B) > μ_β + k*σ_β

### 6. HDBSCAN (データバブル対応)

#### 6.1 コア距離 (式6)
```
cd(B) = d(B, C) + C.nnDist(k)
```
ここで C は B のminPts番目に近いバブル。

#### 6.2 相互到達距離 (式7)
```
d_m(B, C) = max{cd(B), cd(C), d(B, C)}
```

#### 6.3 MST構築
- Prim's アルゴリズム
- 全ペア相互到達距離を計算してMSTを構築

#### 6.4 階層木構築
- MSTエッジを距離でソート
- Union-Findで結合
- linkage行列: `[left_child, right_child, distance, size]`

#### 6.5 凝縮
- min_cluster_size未満のクラスタを「ポイント脱落」として処理
- 凝縮ツリー: `[parent, child, lambda_val, child_size]`

#### 6.6 安定性計算
```
stability(C) = Σ_{p∈C} (λ_p - λ_birth) * child_size
```

#### 6.7 EOMクラスタ選択
- 葉から上向きに処理
- `subtree_stability > cluster.stability` → 子を選択
- それ以外 → 親を選択、子を非選択

#### 6.8 ラベル割り当て
- 選択されたクラスタにラベル (0, 1, 2, ...) を割り当て
- どのクラスタにも属さないポイントは -1 (ノイズ)
- メンバーシップ確率: `prob(p) = λ_p / λ_max`

## 公開API

```rust
/// 動的HDBSCANクラスタリング
pub struct HdbscanIncremental {
    tree: BubbleTree,
    params: HdbscanParams,
    points: Vec<Option<PointEntry>>,
}

/// パラメータ
pub struct HdbscanParams {
    pub min_pts: usize,          // minPts (デフォルト: 100)
    pub min_cluster_size: usize, // 最小クラスタサイズ
    pub cluster_selection_method: ClusterSelection,
    pub compression_rate: f64,   // 圧縮率 (デフォルト: 0.01)
}

/// クラスタ選択方法
pub enum ClusterSelection {
    Eom,   // Excess of Mass (デフォルト)
    Leaf,  // 葉を選択
}

/// クラスタリング結果
pub struct ClusterResult {
    pub labels: Vec<i32>,       // -1 = ノイズ
    pub probabilities: Vec<f64>,
    pub num_clusters: usize,
    pub stability: Vec<f64>,
}

impl HdbscanIncremental {
    /// 新規作成
    pub fn new(dim: usize, params: HdbscanParams) -> Self;
    
    /// ベクトル追加 (ID返却)
    pub fn add(&mut self, vectors: &[Vec<f64>]) -> Vec<usize>;
    
    /// ベクトル削除
    pub fn remove(&mut self, ids: &[usize]);
    
    /// クラスタリング実行
    pub fn cluster(&self) -> ClusterResult;
    
    /// 現在のデータバブル数
    pub fn num_bubbles(&self) -> usize;
    
    /// 現在のポイント数
    pub fn num_points(&self) -> usize;
}
```

## 距離指標

CCIPベクトル（768次元）にはコサイン距離を使用:

```
cosine_distance(a, b) = 1 - (a·b) / (|a| * |b|)
```

## デフォルトパラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `dim` | 768 | CCIPベクトルの次元数 |
| `min_pts` | 100 | HDBSCANの密度パラメータ |
| `min_cluster_size` | 100 | 最小クラスタサイズ |
| `compression_rate` | 0.01 | 圧縮率 (1%) |
| `m` | 25 | Bubble-tree最小ファンアウト |
| `max_fanout` | 50 | Bubble-tree最大ファンアウト |
| `cluster_selection_method` | EOM | クラスタ選択方法 |

## 参照

- 論文: "Dynamic data summarization for hierarchical spatial clustering" (arXiv:2412.07789)
- HDBSCAN: Campello et al., "Density-based clustering based on hierarchical density estimates" (PAKDD 2013)
- BIRCH: Zhang et al., "BIRCH: An efficient data clustering method for very large databases" (1996)
- Data Bubbles: Breunig et al., "Data bubbles: Quality preserving performance boosting for hierarchical clustering" (2001)
