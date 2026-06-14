use crate::data_bubble::DataBubble;
use crate::distance;
use crate::types::{ClusterSelection, ClusterResult, CondensedTreeNode, HdbscanError, HdbscanParams, Label, LinkageRow};

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
        let linkage = self.build_linkage(&mst, n);

        // Step 4: Condense the tree
        let condensed = self.condense_tree(&linkage, min_cluster_size);

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
        let mut core_distances = vec![0.0; n];

        for i in 0..n {
            let mut distances: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| bubbles[i].core_distance(&bubbles[j], min_pts))
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            core_distances[i] = if distances.len() >= min_pts {
                distances[min_pts - 1]
            } else {
                *distances.last().unwrap_or(&0.0)
            };
        }

        core_distances
    }

    fn build_mst(&self, bubbles: &[DataBubble], core_distances: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = bubbles.len();
        let mut mst = Vec::with_capacity(n - 1);
        let mut in_mst = vec![false; n];
        let mut min_edge = vec![f64::INFINITY; n];
        let mut parent = vec![usize::MAX; n];

        in_mst[0] = true;
        for j in 1..n {
            let d = self.mutual_reachability(&bubbles[0], &bubbles[j], core_distances);
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
                    let d = self.mutual_reachability(&bubbles[best], &bubbles[j], core_distances);
                    if d < min_edge[j] {
                        min_edge[j] = d;
                        parent[j] = best;
                    }
                }
            }
        }

        mst
    }

    fn mutual_reachability(&self, a: &DataBubble, b: &DataBubble, core_distances: &[f64]) -> f64 {
        let i = 0; // placeholder - need actual indices
        let j = 0; // placeholder
        let d = distance::cosine_distance(&a.rep, &b.rep);
        d.max(core_distances[i]).max(core_distances[j])
    }

    fn build_linkage(&self, mst: &[(usize, usize, f64)], n: usize) -> Vec<LinkageRow> {
        let mut linkage = Vec::with_capacity(n - 1);
        let mut sorted_mst = mst.to_vec();
        sorted_mst.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        let mut uf = UnionFind::new(2 * n);
        let mut next_label = n;

        for (u, v, dist) in sorted_mst {
            let left = uf.find(u);
            let right = uf.find(v);
            let left_size = self.component_size(&uf, left, n);
            let right_size = self.component_size(&uf, right, n);

            uf.union(left, next_label);
            uf.union(right, next_label);

            linkage.push(LinkageRow {
                left,
                right,
                distance: dist,
                size: left_size + right_size,
            });

            next_label += 1;
        }

        linkage
    }

    fn component_size(&self, _uf: &UnionFind, root: usize, n: usize) -> usize {
        if root < n {
            1
        } else {
            // This is a simplified version - in practice, we'd track sizes
            1
        }
    }

    fn condense_tree(&self, linkage: &[LinkageRow], min_cluster_size: usize) -> Vec<CondensedTreeNode> {
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
                    1
                };

                let right_count = if right >= n {
                    linkage[right - n].size
                } else {
                    1
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
                    self.collect_points(left, &linkage, &mut ignore, &mut result, node, lambda, n);
                    self.collect_points(right, &linkage, &mut ignore, &mut result, node, lambda, n);
                } else if left_count < min_cluster_size {
                    self.collect_points(left, &linkage, &mut ignore, &mut result, node, lambda, n);
                    to_process.push(right);
                } else {
                    self.collect_points(right, &linkage, &mut ignore, &mut result, node, lambda, n);
                    to_process.push(left);
                }
            }
        }

        result
    }

    fn collect_points(
        &self,
        node: usize,
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
                child_size: 1,
            });
        } else {
            ignore[node] = true;
            let row = &linkage[node - n];
            self.collect_points(row.left, linkage, ignore, result, parent, lambda, n);
            self.collect_points(row.right, linkage, ignore, result, parent, lambda, n);
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
        let mut is_cluster: std::collections::HashMap<usize, bool> = std::collections::HashMap::new();
        let mut cluster_stability: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();

        // Initialize all clusters as selected
        for &cluster_id in stability.keys() {
            is_cluster.insert(cluster_id, true);
            cluster_stability.insert(cluster_id, *stability.get(&cluster_id).unwrap_or(&0.0));
        }

        // Process from leaves to root
        let mut nodes: Vec<usize> = stability.keys().cloned().collect();
        nodes.sort_by(|a, b| b.cmp(a)); // Reverse topological sort

        for node in nodes {
            let mut subtree_stability = 0.0;
            for child_node in condensed.iter().filter(|c| c.parent == node) {
                subtree_stability += cluster_stability.get(&child_node.child).unwrap_or(&0.0);
            }

            let node_stability = cluster_stability.get(&node).unwrap_or(&0.0);
            if subtree_stability > *node_stability {
                is_cluster.insert(node, false);
                cluster_stability.insert(node, subtree_stability);
            } else {
                // Unselect all descendants
                for desc in condensed.iter().filter(|c| c.parent == node) {
                    is_cluster.insert(desc.child, false);
                }
            }
        }

        // Assign labels
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

        for node in condensed {
            if node.child < n {
                if let Some(&label) = cluster_map.get(&node.parent) {
                    labels[node.child] = label;
                    let max_lambda = condensed
                        .iter()
                        .filter(|c| c.parent == node.parent)
                        .map(|c| c.lambda_val)
                        .fold(0.0f64, f64::max);
                    if max_lambda > 0.0 {
                        probabilities[node.child] = (node.lambda_val / max_lambda).min(1.0);
                    } else {
                        probabilities[node.child] = 1.0;
                    }
                }
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
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x != root_y {
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
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
