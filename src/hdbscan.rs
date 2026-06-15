use crate::data_bubble::DataBubble;
use crate::distance;
use crate::types::{ClusterResult, ClusterSelection, CondensedTreeNode, HdbscanError, HdbscanParams, Label, LinkageRow};
use rayon::prelude::*;

#[cfg(feature = "turbovec")]
use turbovec::TurboQuantIndex;

pub struct Hdbscan {
    params: HdbscanParams,
}

impl Hdbscan {
    pub fn new(params: HdbscanParams) -> Self {
        Self { params }
    }

    pub fn cluster(&self, bubbles: &[DataBubble]) -> Result<ClusterResult, HdbscanError> {
        if bubbles.is_empty() {
            return Err(HdbscanError::NoPoints);
        }

        let n = bubbles.len();
        let min_pts = self.params.min_pts.min(n.saturating_sub(1)).max(1);
        let min_cluster_size = self.params.min_cluster_size.min(n).max(2);

        // Step 1: Compute core distances
        let core_distances = self.compute_core_distances(bubbles, min_pts);

        // Step 2: Compute mutual reachability distances and build MST
        let mst = self.build_mst(bubbles, &core_distances);

        // Step 3: Build linkage (dendrogram)
        let linkage = self.build_linkage(bubbles, &mst, n);

        // Step 4: Condense the tree
        let condensed = self.condense_tree(bubbles, &linkage, min_cluster_size);

        // Step 5: Compute stability
        let stability = self.compute_stability(&condensed);

        // Step 6: Extract clusters (EOM or Leaf)
        let (labels, probabilities) = match self.params.cluster_selection_method {
            ClusterSelection::Eom => self.extract_clusters_eom(&condensed, &stability, n)?,
            ClusterSelection::Leaf => self.extract_clusters_leaf(&condensed, &stability, n)?,
        };

        let num_clusters = labels.iter().filter(|&&l| l >= 0).map(|&l| l as usize).max().map_or(0, |m| m + 1);

        Ok(ClusterResult {
            labels,
            probabilities,
            num_clusters,
            stability: stability.values().cloned().collect(),
        })
    }

    fn compute_core_distances(&self, bubbles: &[DataBubble], min_pts: usize) -> Vec<f64> {
        let n = bubbles.len();
        if n == 0 {
            return Vec::new();
        }

        #[cfg(feature = "turbovec")]
        if let Some(bit_width) = self.params.turbovec_bit_width {
            if n >= 100 {
                return self.compute_core_distances_turbovec(bubbles, min_pts, bit_width);
            }
        }

        self.compute_core_distances_exhaustive(bubbles, min_pts)
    }

    fn compute_core_distances_exhaustive(&self, bubbles: &[DataBubble], min_pts: usize) -> Vec<f64> {
        let n = bubbles.len();
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut distances: Vec<f64> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| bubbles[i].core_distance(&bubbles[j], min_pts))
                    .collect();
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if distances.len() >= min_pts {
                    distances[min_pts - 1]
                } else {
                    *distances.last().unwrap_or(&0.0)
                }
            })
            .collect()
    }

    #[cfg(feature = "turbovec")]
    fn compute_core_distances_turbovec(
        &self,
        bubbles: &[DataBubble],
        min_pts: usize,
        bit_width: usize,
    ) -> Vec<f64> {
        let n = bubbles.len();
        let dim = bubbles[0].rep.len();

        let mut vectors: Vec<f32> = Vec::with_capacity(n * dim);
        for b in bubbles {
            for &x in &b.rep {
                vectors.push(x as f32);
            }
        }

        let mut index = TurboQuantIndex::new(dim, bit_width);
        index.add(&vectors);
        index.prepare();

        let k = min_pts.min(n.saturating_sub(1)).max(1);
        let search_k = (k + 5).min(n.saturating_sub(1)).max(1);
        let results = index.search(&vectors, search_k);

        (0..n)
            .into_par_iter()
            .map(|i| {
                let candidates = results.indices_for_query(i);
                let mut distances: Vec<f64> = candidates
                    .iter()
                    .filter(|&&j| j != i as i64 && j >= 0 && (j as usize) < n)
                    .map(|&j| bubbles[i].core_distance(&bubbles[j as usize], k))
                    .collect();
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if distances.len() >= k {
                    distances[k - 1]
                } else {
                    *distances.last().unwrap_or(&0.0)
                }
            })
            .collect()
    }

    fn build_mst(&self, bubbles: &[DataBubble], core_distances: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = bubbles.len();
        if n == 0 {
            return Vec::new();
        }
        let mut mst = Vec::with_capacity(n.saturating_sub(1));
        let mut in_mst = vec![false; n];
        let mut min_edge = vec![f64::INFINITY; n];
        let mut parent = vec![usize::MAX; n];

        in_mst[0] = true;
        for j in 1..n {
            let d = self.mutual_reachability(0, j, bubbles, core_distances);
            min_edge[j] = d;
            parent[j] = 0;
        }

        for _ in 0..n - 1 {
            let mut best = usize::MAX;
            let mut best_dist = f64::INFINITY;
            for j in 0..n {
                if !in_mst[j] && min_edge[j] < best_dist {
                    best_dist = min_edge[j];
                    best = j;
                }
            }

            if best == usize::MAX {
                break;
            }

            in_mst[best] = true;
            mst.push((parent[best], best, best_dist));

            for j in 0..n {
                if !in_mst[j] {
                    let d = self.mutual_reachability(best, j, bubbles, core_distances);
                    if d < min_edge[j] {
                        min_edge[j] = d;
                        parent[j] = best;
                    }
                }
            }
        }

        mst
    }

    fn mutual_reachability(&self, i: usize, j: usize, bubbles: &[DataBubble], core_distances: &[f64]) -> f64 {
        let d = distance::cosine_distance(&bubbles[i].rep, &bubbles[j].rep);
        d.max(core_distances[i]).max(core_distances[j])
    }

    fn build_linkage(&self, bubbles: &[DataBubble], mst: &[(usize, usize, f64)], n: usize) -> Vec<LinkageRow> {
        let mut linkage = Vec::with_capacity(n - 1);
        let mut sorted_mst = mst.to_vec();
        sorted_mst.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        let mut weights: Vec<usize> = vec![0; 2 * n];
        for (i, b) in bubbles.iter().enumerate() {
            weights[i] = b.n;
        }
        let mut uf = UnionFind::new(2 * n, &weights);

        for (next_label, (u, v, dist)) in (n..).zip(sorted_mst) {
            let left = uf.find(u);
            let right = uf.find(v);
            let left_weight = uf.component_weight(left);
            let right_weight = uf.component_weight(right);

            uf.union_to_parent(left, next_label);
            uf.union_to_parent(right, next_label);

            linkage.push(LinkageRow {
                left,
                right,
                distance: dist,
                size: left_weight + right_weight,
            });
        }

        linkage
    }

    fn condense_tree(
        &self,
        bubbles: &[DataBubble],
        linkage: &[LinkageRow],
        min_cluster_size: usize,
    ) -> Vec<CondensedTreeNode> {
        let n = linkage.len() + 1;
        let root = 2 * n - 2;
        let mut result = Vec::new();

        // BFS from root
        let mut to_process = vec![root];
        let mut ignore = vec![false; root + 1];

        while !to_process.is_empty() {
            let current_level = to_process.clone();
            to_process.clear();

            for node in current_level {
                if ignore[node] || node < n {
                    continue;
                }

                let row = &linkage[node - n];
                let left = row.left;
                let right = row.right;
                let lambda = if row.distance > 0.0 {
                    1.0 / row.distance
                } else {
                    f64::INFINITY
                };

                let left_count = if left >= n {
                    linkage[left - n].size
                } else {
                    bubbles[left].n
                };

                let right_count = if right >= n {
                    linkage[right - n].size
                } else {
                    bubbles[right].n
                };

                if left_count >= min_cluster_size && right_count >= min_cluster_size {
                    result.push(CondensedTreeNode {
                        parent: node,
                        child: left,
                        lambda_val: lambda,
                        child_size: left_count,
                    });
                    result.push(CondensedTreeNode {
                        parent: node,
                        child: right,
                        lambda_val: lambda,
                        child_size: right_count,
                    });
                    to_process.push(left);
                    to_process.push(right);
                } else if left_count < min_cluster_size && right_count < min_cluster_size {
                    self.collect_points(left, bubbles, linkage, &mut ignore, &mut result, node, lambda, n);
                    self.collect_points(right, bubbles, linkage, &mut ignore, &mut result, node, lambda, n);
                } else if left_count < min_cluster_size {
                    self.collect_points(left, bubbles, linkage, &mut ignore, &mut result, node, lambda, n);
                    to_process.push(right);
                } else {
                    self.collect_points(right, bubbles, linkage, &mut ignore, &mut result, node, lambda, n);
                    to_process.push(left);
                }
            }
        }

        result
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_points(
        &self,
        node: usize,
        bubbles: &[DataBubble],
        linkage: &[LinkageRow],
        ignore: &mut [bool],
        result: &mut Vec<CondensedTreeNode>,
        parent: usize,
        lambda: f64,
        n: usize,
    ) {
        if node < n {
            result.push(CondensedTreeNode {
                parent,
                child: node,
                lambda_val: lambda,
                child_size: bubbles[node].n,
            });
        } else {
            ignore[node] = true;
            let row = &linkage[node - n];
            self.collect_points(row.left, bubbles, linkage, ignore, result, parent, lambda, n);
            self.collect_points(row.right, bubbles, linkage, ignore, result, parent, lambda, n);
        }
    }

    fn compute_stability(&self, condensed: &[CondensedTreeNode]) -> std::collections::HashMap<usize, f64> {
        let mut births: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        let mut stability: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();

        // Find birth lambda for each child
        for node in condensed {
            let entry = births.entry(node.child).or_insert(node.lambda_val);
            if node.lambda_val < *entry {
                *entry = node.lambda_val;
            }
        }

        // Compute stability
        for node in condensed {
            let birth = *births.get(&node.parent).unwrap_or(&0.0);
            let entry = stability.entry(node.parent).or_insert(0.0);
            *entry += (node.lambda_val - birth) * node.child_size as f64;
        }

        stability
    }

    fn extract_clusters_eom(
        &self,
        condensed: &[CondensedTreeNode],
        stability: &std::collections::HashMap<usize, f64>,
        n: usize,
    ) -> Result<(Vec<Label>, Vec<f64>), HdbscanError> {
        if condensed.is_empty() {
            // No clusters found; all points are noise
            return Ok((vec![-1i32; n], vec![0.0f64; n]));
        }

        let mut is_cluster: std::collections::HashMap<usize, bool> = std::collections::HashMap::new();
        let mut cluster_stability: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();

        // Initialize all clusters as selected
        for &cluster_id in stability.keys() {
            is_cluster.insert(cluster_id, true);
            cluster_stability.insert(cluster_id, *stability.get(&cluster_id).unwrap_or(&0.0));
        }

        // Build parent -> child-cluster map for fast lookup
        let mut children_of: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for node in condensed {
            if node.child >= n {
                children_of.entry(node.parent).or_default().push(node.child);
            }
        }

        // Process from leaves to root (reverse order of cluster id)
        let mut nodes: Vec<usize> = stability.keys().cloned().collect();
        nodes.sort_by(|a, b| b.cmp(a));

        for node in nodes {
            let children = children_of.get(&node).cloned().unwrap_or_default();
            let subtree_stability: f64 = children
                .iter()
                .map(|c| cluster_stability.get(c).unwrap_or(&0.0))
                .sum();

            let node_stability = *cluster_stability.get(&node).unwrap_or(&0.0);
            if subtree_stability > node_stability {
                is_cluster.insert(node, false);
                cluster_stability.insert(node, subtree_stability);
            } else {
                for child in children {
                    is_cluster.insert(child, false);
                }
            }
        }

        // Build child -> parent map, keeping only the highest lambda edge for each child
        let mut child_to_parent: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for node in condensed {
            child_to_parent
                .entry(node.child)
                .and_modify(|p| {
                    let existing_lambda = condensed
                        .iter()
                        .find(|c| c.child == *p)
                        .map(|c| c.lambda_val)
                        .unwrap_or(0.0);
                    if node.lambda_val > existing_lambda {
                        *p = node.parent;
                    }
                })
                .or_insert(node.parent);
        }

        // Assign labels by walking from root down to points
        let mut labels = vec![-1i32; n];
        let mut probabilities = vec![0.0f64; n];
        let mut cluster_map = std::collections::HashMap::new();
        let mut cluster_label = 0i32;

        for (&cluster_id, &selected) in &is_cluster {
            if selected {
                cluster_map.insert(cluster_id, cluster_label);
                cluster_label += 1;
            }
        }

        // For each point, find the nearest selected ancestor by lambda
        let mut point_cluster_lambda: std::collections::HashMap<usize, (i32, f64)> = std::collections::HashMap::new();

        for (point, label_slot) in labels.iter_mut().enumerate().take(n) {
            let mut current = point;
            let mut best_cluster: Option<(usize, f64)> = None;
            let mut lambda = f64::INFINITY;

            while let Some(&parent) = child_to_parent.get(&current) {
                // Find the lambda for current under parent
                let edge_lambda = condensed
                    .iter()
                    .find(|c| c.parent == parent && c.child == current)
                    .map(|c| c.lambda_val)
                    .unwrap_or(0.0);
                lambda = lambda.min(edge_lambda);

                if is_cluster.get(&parent).copied().unwrap_or(false) {
                    best_cluster = Some((parent, lambda));
                    break; // nearest selected ancestor found
                }
                current = parent;
            }

            if let Some((cluster_id, lambda_val)) = best_cluster {
                if let Some(&label) = cluster_map.get(&cluster_id) {
                    *label_slot = label;
                    point_cluster_lambda.insert(point, (label, lambda_val));
                }
            }
        }

        // Compute probabilities: lambda_p / max_lambda_in_cluster
        let mut max_lambda_by_cluster: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
        for &(label, lambda_val) in point_cluster_lambda.values() {
            let entry = max_lambda_by_cluster.entry(label).or_insert(lambda_val);
            if lambda_val > *entry {
                *entry = lambda_val;
            }
        }

        for (point, (label, lambda_val)) in point_cluster_lambda {
            let max_lambda = *max_lambda_by_cluster.get(&label).unwrap_or(&lambda_val);
            if max_lambda > 0.0 && lambda_val.is_finite() {
                probabilities[point] = (lambda_val / max_lambda).min(1.0);
            } else {
                probabilities[point] = 1.0;
            }
        }

        Ok((labels, probabilities))
    }

    fn extract_clusters_leaf(
        &self,
        condensed: &[CondensedTreeNode],
        _stability: &std::collections::HashMap<usize, f64>,
        n: usize,
    ) -> Result<(Vec<Label>, Vec<f64>), HdbscanError> {
        // Leaf method: select leaf clusters
        let mut is_cluster: std::collections::HashMap<usize, bool> = std::collections::HashMap::new();
        let mut children_of: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();

        for node in condensed {
            children_of.entry(node.parent).or_default().push(node.child);
        }

        // Find leaves (nodes that are not parents)
        let all_parents: std::collections::HashSet<usize> = children_of.keys().cloned().collect();
        let all_children: std::collections::HashSet<usize> = condensed.iter().map(|c| c.child).collect();
        let leaves: Vec<usize> = all_children.difference(&all_parents).cloned().collect();

        for &leaf in &leaves {
            is_cluster.insert(leaf, true);
        }

        // Assign labels
        let mut labels = vec![-1i32; n];
        let mut probabilities = vec![0.0f64; n];
        let mut cluster_map = std::collections::HashMap::new();
        let mut cluster_label = 0i32;

        for &leaf in &leaves {
            if leaf < n {
                cluster_map.insert(leaf, cluster_label);
                cluster_label += 1;
            }
        }

        for node in condensed {
            if node.child < n {
                if let Some(&label) = cluster_map.get(&node.parent) {
                    labels[node.child] = label;
                    probabilities[node.child] = 1.0;
                }
            }
        }

        Ok((labels, probabilities))
    }
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    weight: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize, weights: &[usize]) -> Self {
        Self {
            parent: (0..size).collect(),
            size: vec![1; size],
            weight: weights.to_vec(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union_to_parent(&mut self, child: usize, parent: usize) {
        let root_child = self.find(child);
        let root_parent = self.find(parent);
        if root_child != root_parent {
            let new_size = self.size[root_child] + self.size[root_parent];
            let new_weight = self.weight[root_child] + self.weight[root_parent];
            self.parent[root_child] = root_parent;
            self.size[root_parent] = new_size;
            self.weight[root_parent] = new_weight;
        }
    }

    fn component_weight(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.weight[root]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdbscan_basic() {
        let params = HdbscanParams {
            min_pts: 2,
            min_cluster_size: 2,
            ..Default::default()
        };
        let hdbscan = Hdbscan::new(params);

        let bubbles = vec![
            DataBubble::from_cf(&crate::cf::ClusteringFeature::from_point(&[0.0, 0.0])),
            DataBubble::from_cf(&crate::cf::ClusteringFeature::from_point(&[0.1, 0.1])),
            DataBubble::from_cf(&crate::cf::ClusteringFeature::from_point(&[10.0, 10.0])),
            DataBubble::from_cf(&crate::cf::ClusteringFeature::from_point(&[10.1, 10.1])),
        ];

        let result = hdbscan.cluster(&bubbles).unwrap();
        assert!(result.num_clusters >= 1);
    }
}
