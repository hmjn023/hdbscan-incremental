# hdbscan-incremental - 設計仕様

## 位置づけ

この crate は、動的に増減するベクトル集合に対して HDBSCAN 系の階層クラスタリングを現実的な計算量で提供する Rust ライブラリである。

本実装の中心は、Kayumov, Kim, Shin "Dynamic data summarization for hierarchical spatial clustering" (2024) の第4章にある online/offline 構成である。第3章の「厳密に動的 MST を保守する HDBSCAN」は研究上の比較対象であり、この crate の主要実装対象ではない。理由は同論文が示す通り、挿入・削除により reverse kNN の core distance と mutual reachability edge が連鎖的に変わり、少量の更新でも静的再計算と同程度の仕事量になり得るためである。

したがって、本 crate は次を目標にする。

1. online phase で Bubble-tree を保守し、現在の N 点を L 個前後の Clustering Feature (CF) に圧縮する。
2. cluster 要求時に leaf CF を Data Bubble に変換する。
3. Data Bubble を重み付き点として静的 HDBSCAN の階層構築・凝縮・クラスタ選択へ渡す。

厳密な点単位 HDBSCAN と同一結果を保証する実装ではない。設計上の品質目標は、静的 HDBSCAN に近いクラスタリング品質を、動的更新と圧縮により実用的な時間・メモリで得ることである。

## 参照設計

### 一次資料

- Dynamic data summarization for hierarchical spatial clustering, arXiv:2412.07789, https://arxiv.org/abs/2412.07789
- Campello, Moulavi, Sander, "Density-Based Clustering Based on Hierarchical Density Estimates", PAKDD 2013, https://doi.org/10.1007/978-3-642-37456-2_14
- Campello, Moulavi, Zimek, Sander, "Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection", TKDD 2015, https://doi.org/10.1145/2733381
- Zhang, Ramakrishnan, Livny, "BIRCH: An Efficient Data Clustering Method for Very Large Databases", SIGMOD 1996
- Breunig, Kriegel, Kroger, Pfeifle, "Data Bubbles: Quality Preserving Performance Boosting for Hierarchical Clustering", SIGMOD Record 2001
- Schubert, Lang, "BETULA: Numerically Stable CF-Trees for BIRCH Clustering", SISAP 2020, https://arxiv.org/abs/2006.12881

### 実装上の解釈

元論文の HDBSCAN 背景説明はユークリッド距離を前提にしている。この crate は CCIP などの高次元埋め込み利用を主対象にするため、距離関数は設定可能にする。既定値は cosine distance とする。ただし Data Bubble の `extent` と `nnDist(k)` はユークリッド幾何を前提にした近似式なので、cosine distance 利用時は近似として扱い、仕様上も「厳密な幾何的半径」ではなく「密度補正項」として扱う。

## 公開 API

```rust
pub struct HdbscanIncremental { ... }

impl HdbscanIncremental {
    pub fn new(dim: usize, params: HdbscanParams) -> Result<Self, HdbscanError>;
    pub fn add(&mut self, vectors: &[Vec<f64>]) -> Result<Vec<usize>, HdbscanError>;
    pub fn remove(&mut self, ids: &[usize]) -> Result<(), HdbscanError>;
    pub fn cluster(&self) -> Result<ClusterResult, HdbscanError>;
    pub fn num_bubbles(&self) -> usize;
    pub fn num_points(&self) -> usize;
}
```

`new` は不正パラメータを `HdbscanError::InvalidParameter` として返す。既存 API 互換を維持する必要がある間は `new_unchecked` ではなく `try_new` を追加して段階移行してもよい。

### パラメータ

```rust
pub struct HdbscanParams {
    pub min_pts: usize,
    pub min_cluster_size: usize,
    pub cluster_selection_method: ClusterSelection,
    pub compression_rate: f64,
    pub m: usize,
    pub max_fanout: Option<usize>,
    pub distance_metric: DistanceMetric,
    pub turbovec_bit_width: Option<usize>,
}
```

制約:

- `dim > 0`
- `min_pts >= 1`
- `min_cluster_size >= 2`
- `0.0 < compression_rate <= 1.0`
- `m >= 1`
- `max_fanout.unwrap_or(2 * m) >= 2 * m - 1`
- 実効 leaf 目標数 `L = ceil(N * compression_rate)` とし、下限は 1。N が更新されるため L は固定値ではなく現在点数から再計算する。

注: 既存実装は `L = ceil(1 / compression_rate)` としているが、これは「圧縮率 1% なら 100 leaf」という固定 leaf 数になり、論文の「N を L 個へ、例: N の 1%」という意味から外れる。正しくは `target_leaves(N) = ceil(N * compression_rate)` である。

## データモデル

### Point Store

削除を正しく行うため、各挿入点を安定 ID で保持する。

```rust
struct PointEntry {
    id: usize,
    vector: Vec<f64>,
    leaf_id: LeafId,
    generation: u64,
}
```

`leaf_id` は削除時の高速化に使う。Bubble-tree 再編成で点が移動した場合は更新する。再編成実装が単純な間は leaf 直行ではなく fallback search を許容するが、仕様上は O(tree height) 削除を目標にする。

### Clustering Feature

BIRCH の CF を基礎にする。

```rust
pub struct ClusteringFeature {
    pub ls: Vec<f64>,
    pub ss: f64,
    pub n: usize,
}
```

意味:

- `LS = sum_i x_i`
- `SS = sum_i ||x_i||^2`
- `n = |P|`

加法性:

```text
CF(A union B) = {LS_A + LS_B, SS_A + SS_B, n_A + n_B}
```

不変条件:

- `n == 0` の CF を通常 leaf として保持しない。
- `ls.len() == dim`
- `ss >= 0`
- `centroid = LS / n`

数値安定性:

`SS - LS^2 / n` 型の計算は高次元・大 N で桁落ちしやすい。初期実装は BIRCH 互換の `{LS, SS, n}` でよいが、`extent` が負値に落ちた場合は 0 に clamp し、将来の改善として BETULA 型 `{n, mean, sum_squared_deviation}` へ移行可能な内部 trait を切る。

### Data Bubble

Data Bubble は leaf CF から派生する静的クラスタリング用の重み付き代表点である。

```rust
pub struct DataBubble {
    pub rep: Vec<f64>,
    pub n: usize,
    pub extent: f64,
}
```

式:

```text
rep = LS / n
extent = sqrt(max(0, (2 * n * SS - 2 * ||LS||^2) / (n * (n - 1))))
nnDist(k) = (k / n)^(1 / dim) * extent
```

境界条件:

- `n == 1` の `extent` は 0。
- `k == 0` の `nnDist` は 0。
- `k >= n` の扱いは `nnDist(n)` に clamp する。Data Bubble 内部の点数を超える近傍距離を外挿しない。

## Bubble-tree

Bubble-tree は fully dynamic な CF 木であり、leaf が実点集合を表す。internal node は子 node の CF 合計を保持する。

### 構造

```rust
pub struct BubbleTree {
    root: Node,
    dim: usize,
    min_fanout: usize,  // m
    max_fanout: usize,  // M
    compression_rate: f64,
    total_n: usize,
}
```

不変条件:

- root が leaf の場合は全点を持つ単一 leaf としてよい。
- root が internal の場合、子数は 2..=M。
- root 以外の internal node の子数は m..=M。
- leaf は 1 個以上の点を保持する。
- すべての internal CF は子 CF の合計と一致する。
- `num_leaves` は可能な限り `target_leaves(total_n)` に近づける。

論文の Property 4 は「leaf 数を L に保つ」と書くが、N が小さい場合、`L > N` の場合、`m` 制約により分割不能な場合がある。実装仕様では次を採用する。

```text
target = clamp(ceil(total_n * compression_rate), 1, total_n)
acceptable if:
  num_leaves == target
  or no legal split/merge/reinsert operation can move num_leaves closer to target
```

### 挿入

1. root から leaf まで、各階層で `distance(point, child.centroid)` が最小の子を選ぶ。
2. leaf に点を追加し、leaf CF を更新する。
3. 祖先 CF を leaf から root まで更新する。
4. leaf または internal node が `max_fanout` を超えた場合は split する。
5. `maintain_compression()` を呼ぶ。

### 削除

1. ID から対象点と候補 leaf を得る。
2. leaf から点を削除し、祖先 CF を更新する。
3. 空 leaf は削除する。
4. root 以外の node が `min_fanout` を下回る場合は、node を削除してその子または点を再挿入する。
5. root が単一 internal child だけを持つ場合は height を縮める。
6. `maintain_compression()` を呼ぶ。

削除は「同じ座標の最初の点」を消すのではなく、挿入 ID に対応する点を消す。重複ベクトルは合法である。

### Split

node の子要素または leaf 内の点を 2 群に分ける。

1. 全ペア距離が最大の 2 要素を seed にする。
2. 残りを近い seed 側へ割り当てる。
3. どちらかの群が `m` 未満なら、遠い順に反対群から移して `m` を満たす。
4. 両群の CF を再計算する。

leaf split では点を分割する。internal split では子 node を分割し、parent pointer を更新する。

### Compression Maintenance

元論文 Algorithm 1 に基づく。

```text
target = target_leaves(total_n)

if num_leaves > target:
    choose the most under-filled leaf U
    remove U
    reinsert U's points

else if num_leaves < target:
    choose the most over-filled splittable leaf O
    split O to create sibling O'
    reinsert O' or attach it through normal tree insertion

else:
    choose the most over-filled leaf O
    extract m farthest points/children of O
    reinsert them
```

under-filled / over-filled 判定:

```text
beta(B) = B.n / total_n
mu = mean(beta over leaves)
sigma = stddev(beta over leaves)
good if beta in [mu - k * sigma, mu + k * sigma]
under-filled if beta < mu - k * sigma
over-filled if beta > mu + k * sigma
```

実装では Chebyshev パラメータ `k` を設定値にする。既定値は 2.0。単純な最大/最小 leaf サイズ選択は fallback として許容するが、仕様上は beta に基づく選択を標準にする。

再挿入中に `maintain_compression()` が再帰して無限ループしないよう、内部 API は `insert_raw` と `rebalance_once` を分ける。

## Data Bubble HDBSCAN

### Core Distance

点単位 HDBSCAN の core distance は `minPts` 番目の近傍距離である。Data Bubble では bubble が複数点を代表するため、近傍探索は bubble weight を累積して行う。

bubble `B` の core distance:

1. 他 bubble `C` を `distance(B.rep, C.rep)` 昇順に見る。
2. `C` より近い bubble の重み合計を `w_before` とする。
3. `w_before + k` が `min_pts` 個の原点を表すように `k = min(min_pts - w_before, C.n)` を選ぶ。
4. `cd(B) = distance(B.rep, C.rep) + C.nnDist(k)`。

この仕様では、現在実装のように「bubble 個数に対する `min_pts` 番目」を core distance としない。`min_pts` は原点数スケールの密度パラメータであり、bubble 数スケールに変換してはならない。

### Mutual Reachability

```text
mreach(B, C) = max(cd(B), cd(C), distance(B.rep, C.rep))
```

### MST

初期実装は全ペア距離をオンデマンド計算する Prim でよい。

計算量:

```text
L = num_bubbles
core distances: O(L^2 log L) naive, O(L^2) with selection
MST: O(L^2)
memory: O(L) if distances are computed on demand, O(L^2) if cached
```

`turbovec` は近似 kNN 候補生成として使ってよいが、近似結果を使う場合は `ClusterResult` または params に approximate であることを明示できる設計にする。既定は exact exhaustive。

### Hierarchy

HDBSCAN は mutual reachability graph の MST から single-linkage hierarchy を作る。

実装手順:

1. MST edge を重み昇順で処理する。
2. Union-Find で connected component を merge し、linkage row を生成する。
3. component size は bubble 数ではなく bubble weight 合計を使う。

linkage row:

```rust
struct LinkageRow {
    left: NodeId,
    right: NodeId,
    distance: f64,
    size: usize, // sum of bubble.n
}
```

### Condensed Tree

`min_cluster_size` も原点数スケールで扱う。

階層を上から辿り、split 時に:

- 両子が `min_cluster_size` 以上なら true split として condensed tree に両子を残す。
- 片方だけ小さいなら、小さい側は parent から落ちた点または bubble 集合として記録し、大きい側は parent identity を継続する。
- 両方小さいなら両方を落下として記録する。

Data Bubble を leaf として扱う場合でも、落下イベントの `child_size` は `bubble.n` を使う。

### Stability と EOM

```text
lambda = 1 / distance
stability(C) = sum_{p in C} (lambda_p - lambda_birth(C))
```

Data Bubble では点 `p` の代わりに bubble weight を掛ける。

```text
stability(C) = sum_{event e under C} (lambda_e - lambda_birth(C)) * event_weight
```

EOM:

1. condensed tree の leaf cluster を候補にする。
2. 逆トポロジ順で子 stability 合計と親自身の stability を比較する。
3. 子合計が大きい場合は親を非選択にし、親の有効 stability を子合計に置き換える。
4. 親自身が大きい場合は親を選択し、すべての子孫候補を非選択にする。
5. root は出力クラスタにしない。

Leaf selection:

- condensed tree 上で子 cluster を持たない cluster node を選択する。
- singleton bubble は `min_cluster_size` を満たす場合だけ cluster 候補になる。

### Label Assignment

この crate の `ClusterResult.labels` は入力点 ID 順の長さ `num_points_ever_inserted` ではなく、現在有効な点の挿入順ビューに対する長さにするか、ID 対応を返す必要がある。仕様上は次を推奨する。

```rust
pub struct ClusterResult {
    pub assignments: Vec<PointAssignment>,
    pub num_clusters: usize,
    pub cluster_stability: Vec<f64>,
}

pub struct PointAssignment {
    pub id: usize,
    pub label: i32,
    pub probability: f64,
}
```

互換 API として `labels: Vec<i32>` を残す場合は、現在有効な点だけを `active_ids()` 昇順に並べた結果であることを明記する。

Data Bubble 上で選択された cluster は、その bubble に属する全点へ同じ label を割り当てる。bubble 内部での点単位境界は復元しない。

probability:

```text
probability(p) = clamp(lambda_p / max_lambda_selected_cluster, 0, 1)
```

Data Bubble 内点は bubble の落下 lambda を共有する。より細かい確率が必要な場合は、bubble 内点と bubble centroid の距離で補正するオプションを別途設計する。

## 距離関数

```rust
pub enum DistanceMetric {
    Cosine,
    Euclidean,
}
```

既定は `Cosine`。

制約:

- zero vector 同士の cosine distance は 0 とする。
- zero vector と非 zero vector の cosine distance は 1 とする。
- cosine distance は `[0, 2]` の範囲に丸める。

Data Bubble の extent は元式に従いユークリッド的に計算する。`DistanceMetric::Cosine` で使う場合は近似である。

## エラー処理

fallible 操作は `Result<_, HdbscanError>` を返す。

```rust
pub enum HdbscanError {
    InvalidDimension { expected: usize, actual: usize },
    PointNotFound(usize),
    NoPoints,
    InvalidParameter(String),
}
```

panic してよいのはテストと内部不変条件違反だけである。ユーザー入力、パラメータ、削除 ID、ベクトル次元は error にする。

## 実装ロードマップ

### Phase 1: 仕様と現実の整合

- `compression_rate` から `target_leaves(N)` を計算する。
- `new` または `try_new` でパラメータ検証する。
- zero vector cosine distance の仕様を固定する。
- `ClusterResult` が bubble 数ではなく点 ID に対応するよう設計を直す。

### Phase 2: Bubble-tree の不変条件

- root/leaf/internal の空 CF をなくす。
- `max_fanout` を実際に enforce する。
- leaf split と internal split を分離する。
- delete by ID を導入し、重複ベクトル削除を正しくする。
- under-filled / over-filled を beta 指標で選ぶ。
- reinsertion 中の compression 再帰を止める。

### Phase 3: Data Bubble HDBSCAN の正確化

- core distance を bubble weight 累積で計算する。
- `min_pts` と `min_cluster_size` を原点数スケールに統一する。
- condensed tree の cluster identity 継続を実装する。
- stability を weighted event として計算する。
- EOM で root を選択しない。

### Phase 4: 品質・性能

- exhaustive 実装を基準として固定する。
- `turbovec` は近似候補生成の feature として分離する。
- ベンチマークで N, L, dim, compression_rate ごとの更新時間と clustering 時間を測る。
- 静的 HDBSCAN 実装または Python hdbscan との NMI/ARI 比較を追加する。

## テスト方針

### CF / Data Bubble

- CF 加法性。
- remove 後に元 CF へ戻る。
- `extent` の single point / duplicate points / separated points。
- `extent` の負値丸め。
- `nnDist(k)` の `k == 0`, `k == n`, `k > n`。

### Bubble-tree

- 挿入後に root CF が全点 CF と一致する。
- 削除後に root CF が残点 CF と一致する。
- 同一ベクトルを複数挿入し、指定 ID だけ削除できる。
- `num_leaves` が target に近づく。
- すべての parent pointer が正しい。
- internal fanout が m..=M を満たす。
- reinsertion が無限再帰しない。

### HDBSCAN

- bubble weight を考慮した core distance。
- `min_cluster_size` が bubble 個数ではなく原点数で効く。
- MST edge 数が `L - 1`。
- condensed tree で小クラスタが落下イベントになる。
- EOM が root ではなく安定な子クラスタを選ぶ。
- label が入力点 ID に戻る。

### 統合

- 2D の二群データで 2 cluster を検出する。
- bridge noise を入れても単一連結へ潰れにくい。
- 挿入順を変えても大きく結果が変わらない。
- 挿入・削除後にクラスタ結果が静的再構築と近い。
- `--features turbovec` でも exhaustive fallback と同じ shape の結果を返す。

## 既知の設計リスク

- Data Bubble の `extent` / `nnDist` はユークリッド空間由来であり、cosine distance では近似になる。
- 高次元では nearest neighbor の距離差が小さくなり、Bubble-tree の最近 centroid 選択が不安定になり得る。
- Bubble 内部の点分布は CF だけでは復元できないため、非球形クラスタ境界では label が粗くなる。
- 圧縮率を低くしすぎると HDBSCAN の階層が leaf CF の配置に支配される。
- `{LS, SS, n}` は数値安定性が弱いため、大 N では BETULA 型への移行を検討する。

## 完了条件

この仕様に対する最小実装完了は次を満たすこと。

- `rtk cargo test` が通る。
- `rtk cargo test --features turbovec` が通る。
- `rtk cargo fmt --check` が通る。
- `rtk cargo clippy --all-targets --all-features` が通る。
- Bubble-tree の CF 不変条件を検査するテストがある。
- Data Bubble HDBSCAN が bubble weight を使うテストがある。
- `README.md` が「近似であること」と `compression_rate` の意味を正しく説明している。
