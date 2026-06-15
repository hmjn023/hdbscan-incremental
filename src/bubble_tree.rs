use crate::cf::ClusteringFeature;
use crate::distance;
use crate::types::Vector;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone)]
pub struct BubbleTree {
    root: Rc<RefCell<Node>>,
    m: usize,
    #[allow(dead_code)]
    // TODO: enforce max_fanout when splitting internal nodes
    max_fanout: usize,
    l: usize,
    dim: usize,
    total_n: usize,
    num_leaves: usize,
}

#[derive(Debug, Clone)]
pub enum Node {
    Internal {
        cf: ClusteringFeature,
        children: Vec<Rc<RefCell<Node>>>,
        parent: Option<Weak<RefCell<Node>>>,
    },
    Leaf {
        cf: ClusteringFeature,
        points: Vec<Vector>,
        parent: Option<Weak<RefCell<Node>>>,
    },
}

impl BubbleTree {
    pub fn new(dim: usize, l: usize, m: usize) -> Self {
        let max_fanout = 2 * m;
        let root = Rc::new(RefCell::new(Node::Leaf {
            cf: ClusteringFeature::from_point(&vec![0.0; dim]),
            points: Vec::new(),
            parent: None,
        }));
        Self {
            root,
            m,
            max_fanout,
            l,
            dim,
            total_n: 0,
            num_leaves: 1,
        }
    }

    pub fn insert(&mut self, point: &[f64]) {
        let best_leaf = self.find_best_leaf(point);
        self.insert_into_leaf(best_leaf, point);
        self.total_n += 1;
        self.maintain_compression();
    }

    fn insert_without_maintain(&mut self, point: &[f64]) {
        let best_leaf = self.find_best_leaf(point);
        self.insert_into_leaf(best_leaf, point);
    }

    pub fn delete(&mut self, point: &[f64]) -> bool {
        let result = self.delete_from_tree(self.root.clone(), point);
        if result {
            self.total_n -= 1;
            self.maintain_compression();
        }
        result
    }

    pub fn extract_leaves(&self) -> Vec<ClusteringFeature> {
        let mut leaves = Vec::new();
        self.collect_leaves(&self.root.borrow(), &mut leaves);
        leaves
    }

    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    pub fn total_points(&self) -> usize {
        self.total_n
    }

    fn find_best_leaf(&self, point: &[f64]) -> Rc<RefCell<Node>> {
        let mut current = self.root.clone();
        loop {
            let next = {
                let node = current.borrow();
                match &*node {
                    Node::Internal { children, .. } => {
                        let mut best_child = None;
                        let mut best_dist = f64::INFINITY;
                        for child in children {
                            let child_borrow = child.borrow();
                            let centroid = child_borrow.cf().centroid();
                            let dist = distance::cosine_distance(&centroid, point);
                            if dist < best_dist {
                                best_dist = dist;
                                best_child = Some(child.clone());
                            }
                        }
                        best_child
                    }
                    Node::Leaf { .. } => None,
                }
            };
            match next {
                Some(child) => current = child,
                None => return current,
            }
        }
    }

    fn insert_into_leaf(&mut self, leaf: Rc<RefCell<Node>>, point: &[f64]) {
        {
            let mut node = leaf.borrow_mut();
            match &mut *node {
                Node::Leaf { cf, points, .. } => {
                    cf.add_point(point);
                    points.push(point.to_vec());
                }
                Node::Internal { .. } => unreachable!(),
            }
        }
        self.update_ancestors_cf(&leaf);
    }

    fn delete_from_tree(&self, node: Rc<RefCell<Node>>, point: &[f64]) -> bool {
        let mut node_mut = node.borrow_mut();
        match &mut *node_mut {
            Node::Leaf { cf, points, .. } => {
                if let Some(pos) = points.iter().position(|p| vectors_equal(p, point)) {
                    cf.remove_point(point);
                    points.remove(pos);
                    return true;
                }
                false
            }
            Node::Internal { cf, children, .. } => {
                for child in children {
                    if self.delete_from_tree(child.clone(), point) {
                        cf.remove_point(point);
                        return true;
                    }
                }
                false
            }
        }
    }

    fn maintain_compression(&mut self) {
        if self.num_leaves > self.l {
            self.remove_underfilled_leaf();
        } else if self.num_leaves < self.l {
            self.split_overfilled_leaf();
        } else {
            self.reorganize();
        }
    }

    fn remove_underfilled_leaf(&mut self) {
        let leaf = self.find_most_underfilled_leaf();
        if let Some(leaf) = leaf {
            let points = {
                let node = leaf.borrow();
                match &*node {
                    Node::Leaf { points, .. } => points.clone(),
                    _ => return,
                }
            };
            self.remove_node_from_parent(leaf);
            self.num_leaves -= 1;
            for p in points {
                self.insert(&p);
            }
        }
    }

    fn split_overfilled_leaf(&mut self) {
        let leaf = self.find_most_overfilled_leaf();
        if let Some(leaf) = leaf {
            let (points, parent) = {
                let node = leaf.borrow();
                match &*node {
                    Node::Leaf { points, parent, .. } => (points.clone(), parent.clone()),
                    _ => return,
                }
            };

            if points.len() < 2 * self.m {
                return;
            }

            let (seed_a, seed_b) = self.find_farthest_pair(&points);
            let (mut group_a, mut group_b) = (Vec::new(), Vec::new());
            for p in &points {
                let dist_a = distance::cosine_distance(p, &points[seed_a]);
                let dist_b = distance::cosine_distance(p, &points[seed_b]);
                if dist_a < dist_b {
                    group_a.push(p.clone());
                } else {
                    group_b.push(p.clone());
                }
            }

            let cf_a = self.create_cf_from_points(&group_a);
            let cf_b = self.create_cf_from_points(&group_b);

            let new_leaf_a = Rc::new(RefCell::new(Node::Leaf {
                cf: cf_a,
                points: group_a,
                parent: parent.clone(),
            }));
            let new_leaf_b = Rc::new(RefCell::new(Node::Leaf {
                cf: cf_b,
                points: group_b,
                parent: parent.clone(),
            }));

            self.remove_node_from_parent(leaf);
            self.num_leaves -= 1;

            if let Some(parent_weak) = parent {
                if let Some(parent_rc) = parent_weak.upgrade() {
                    let mut parent_mut = parent_rc.borrow_mut();
                    match &mut *parent_mut {
                        Node::Internal { children, .. } => {
                            children.push(new_leaf_a.clone());
                            children.push(new_leaf_b.clone());
                            self.num_leaves += 2;
                        }
                        _ => unreachable!(),
                    }
                    drop(parent_mut);
                    self.update_ancestors_cf(&parent_rc);
                }
            } else {
                // The leaf was the root; create a new root
                let new_root_cf = self.merge_cfs(&[
                    new_leaf_a.borrow().cf().clone(),
                    new_leaf_b.borrow().cf().clone(),
                ]);
                let new_root = Rc::new(RefCell::new(Node::Internal {
                    cf: new_root_cf,
                    children: vec![new_leaf_a.clone(), new_leaf_b.clone()],
                    parent: None,
                }));
                {
                    let mut leaf_a_mut = new_leaf_a.borrow_mut();
                    match &mut *leaf_a_mut {
                        Node::Leaf { parent, .. } => *parent = Some(Rc::downgrade(&new_root)),
                        _ => unreachable!(),
                    }
                }
                {
                    let mut leaf_b_mut = new_leaf_b.borrow_mut();
                    match &mut *leaf_b_mut {
                        Node::Leaf { parent, .. } => *parent = Some(Rc::downgrade(&new_root)),
                        _ => unreachable!(),
                    }
                }
                self.root = new_root;
                self.num_leaves += 2;
            }
        }
    }

    fn reorganize(&mut self) {
        let leaf = self.find_most_overfilled_leaf();
        if let Some(leaf) = leaf {
            let points = {
                let node = leaf.borrow();
                match &*node {
                    Node::Leaf { points, .. } => points.clone(),
                    _ => return,
                }
            };

            if points.len() <= self.m {
                return;
            }

            let centroid = self.create_cf_from_points(&points).centroid();
            let mut indexed_distances: Vec<(usize, f64)> = points
                .iter()
                .enumerate()
                .map(|(i, p)| (i, distance::cosine_distance(&centroid, p)))
                .collect();
            indexed_distances
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let farthest_indices: std::collections::HashSet<usize> = indexed_distances
                .iter()
                .take(self.m)
                .map(|(i, _)| *i)
                .collect();

            let remaining: Vec<Vector> = points
                .iter()
                .enumerate()
                .filter(|(i, _)| !farthest_indices.contains(i))
                .map(|(_, p)| p.clone())
                .collect();

            let reinsert: Vec<Vector> = points
                .iter()
                .enumerate()
                .filter(|(i, _)| farthest_indices.contains(i))
                .map(|(_, p)| p.clone())
                .collect();

            {
                let mut node = leaf.borrow_mut();
                match &mut *node {
                    Node::Leaf { cf, points, .. } => {
                        *cf = self.create_cf_from_points(&remaining);
                        *points = remaining;
                    }
                    _ => unreachable!(),
                }
            }
            self.update_ancestors_cf(&leaf);

            for p in reinsert {
                self.insert_without_maintain(&p);
            }
        }
    }

    fn find_most_underfilled_leaf(&self) -> Option<Rc<RefCell<Node>>> {
        let mut best = None;
        let mut best_n = usize::MAX;
        self.find_leaves_recursive(&self.root, &mut |leaf| {
            let n = leaf.borrow().cf().n;
            if n < best_n {
                best_n = n;
                best = Some(leaf.clone());
            }
        });
        best
    }

    fn find_most_overfilled_leaf(&self) -> Option<Rc<RefCell<Node>>> {
        let mut best = None;
        let mut best_n = 0;
        self.find_leaves_recursive(&self.root, &mut |leaf| {
            let n = leaf.borrow().cf().n;
            if n > best_n {
                best_n = n;
                best = Some(leaf.clone());
            }
        });
        best
    }

    fn find_leaves_recursive<F>(&self, node: &Rc<RefCell<Node>>, callback: &mut F)
    where
        F: FnMut(&Rc<RefCell<Node>>),
    {
        let node_borrow = node.borrow();
        match &*node_borrow {
            Node::Leaf { .. } => {
                callback(node);
            }
            Node::Internal { children, .. } => {
                for child in children {
                    let child_borrow = child.borrow();
                    match &*child_borrow {
                        Node::Leaf { .. } => callback(child),
                        Node::Internal { .. } => {
                            drop(child_borrow);
                            self.find_leaves_recursive(child, callback);
                        }
                    }
                }
            }
        }
    }

    fn find_farthest_pair(&self, points: &[Vector]) -> (usize, usize) {
        let mut max_dist = 0.0;
        let mut pair = (0, 1);
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let d = distance::cosine_distance(&points[i], &points[j]);
                if d > max_dist {
                    max_dist = d;
                    pair = (i, j);
                }
            }
        }
        pair
    }

    fn create_cf_from_points(&self, points: &[Vector]) -> ClusteringFeature {
        if points.is_empty() {
            return ClusteringFeature::from_point(&vec![0.0; self.dim]);
        }
        let mut cf = ClusteringFeature::from_point(&points[0]);
        for p in &points[1..] {
            cf.add_point(p);
        }
        cf
    }

    fn merge_cfs(&self, cfs: &[ClusteringFeature]) -> ClusteringFeature {
        if cfs.is_empty() {
            return ClusteringFeature::from_point(&vec![0.0; self.dim]);
        }
        let mut result = cfs[0].clone();
        for cf in &cfs[1..] {
            result = result.merge(cf);
        }
        result
    }

    fn update_ancestors_cf(&self, node: &Rc<RefCell<Node>>) {
        let mut current_weak = {
            let node_borrow = node.borrow();
            match &*node_borrow {
                Node::Internal { parent, .. } => parent.clone(),
                Node::Leaf { parent, .. } => parent.clone(),
            }
        };

        while let Some(parent_weak) = current_weak {
            if let Some(parent_rc) = parent_weak.upgrade() {
                let next_weak = {
                    let parent_borrow = parent_rc.borrow();
                    match &*parent_borrow {
                        Node::Internal { parent, .. } => parent.clone(),
                        _ => None,
                    }
                };

                {
                    let mut parent_mut = parent_rc.borrow_mut();
                    match &mut *parent_mut {
                        Node::Internal { cf, children, .. } => {
                            let child_cfs: Vec<ClusteringFeature> =
                                children.iter().map(|c| c.borrow().cf().clone()).collect();
                            *cf = self.merge_cfs(&child_cfs);
                        }
                        _ => break,
                    }
                }

                current_weak = next_weak;
            } else {
                break;
            }
        }
    }

    fn remove_node_from_parent(&self, node: Rc<RefCell<Node>>) {
        let parent_weak = {
            let node_borrow = node.borrow();
            match &*node_borrow {
                Node::Leaf { parent, .. } => parent.clone(),
                Node::Internal { parent, .. } => parent.clone(),
            }
        };

        if let Some(parent_weak) = parent_weak {
            if let Some(parent_rc) = parent_weak.upgrade() {
                {
                    let mut parent_mut = parent_rc.borrow_mut();
                    match &mut *parent_mut {
                        Node::Internal { children, .. } => {
                            children.retain(|c| !Rc::ptr_eq(c, &node));
                        }
                        _ => unreachable!(),
                    }
                }
                self.update_ancestors_cf(&parent_rc);
            }
        }
    }

    fn collect_leaves(&self, node: &Node, leaves: &mut Vec<ClusteringFeature>) {
        match node {
            Node::Leaf { cf, .. } => leaves.push(cf.clone()),
            Node::Internal { children, .. } => {
                for child in children {
                    self.collect_leaves(&child.borrow(), leaves);
                }
            }
        }
    }
}

impl Node {
    pub fn cf(&self) -> &ClusteringFeature {
        match self {
            Node::Internal { cf, .. } => cf,
            Node::Leaf { cf, .. } => cf,
        }
    }
}

fn vectors_equal(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bubble_tree_insert() {
        let mut tree = BubbleTree::new(3, 2, 1);
        tree.insert(&[1.0, 2.0, 3.0]);
        assert_eq!(tree.total_points(), 1);
        assert_eq!(tree.num_leaves(), 1);
    }

    #[test]
    fn test_bubble_tree_insert_multiple() {
        let mut tree = BubbleTree::new(3, 3, 1);
        tree.insert(&[1.0, 0.0, 0.0]);
        tree.insert(&[0.0, 1.0, 0.0]);
        tree.insert(&[0.0, 0.0, 1.0]);
        assert_eq!(tree.total_points(), 3);
    }

    #[test]
    fn test_bubble_tree_delete() {
        let mut tree = BubbleTree::new(3, 2, 1);
        tree.insert(&[1.0, 2.0, 3.0]);
        tree.insert(&[4.0, 5.0, 6.0]);
        assert_eq!(tree.total_points(), 2);

        let deleted = tree.delete(&[1.0, 2.0, 3.0]);
        assert!(deleted);
        assert_eq!(tree.total_points(), 1);
    }

    #[test]
    fn test_extract_leaves() {
        let mut tree = BubbleTree::new(3, 2, 1);
        tree.insert(&[1.0, 2.0, 3.0]);
        tree.insert(&[4.0, 5.0, 6.0]);
        let leaves = tree.extract_leaves();
        assert!(!leaves.is_empty());
    }
}
