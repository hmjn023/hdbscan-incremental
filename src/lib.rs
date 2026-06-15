#[cfg(all(feature = "turbovec", any(target_os = "linux", target_os = "macos")))]
extern crate blas_src;

pub mod bubble_tree;
pub mod cf;
pub mod data_bubble;
pub mod distance;
pub mod hdbscan;
pub mod types;

pub use types::{ClusterResult, ClusterSelection, HdbscanError, HdbscanParams};

use data_bubble::DataBubble;
use hdbscan::Hdbscan;
use types::PointEntry;

pub struct HdbscanIncremental {
    tree: bubble_tree::BubbleTree,
    params: HdbscanParams,
    points: Vec<Option<PointEntry>>,
    dim: usize,
}

impl HdbscanIncremental {
    pub fn new(dim: usize, params: HdbscanParams) -> Self {
        Self::try_new(dim, params).expect("invalid HdbscanIncremental parameters")
    }

    pub fn try_new(dim: usize, params: HdbscanParams) -> Result<Self, HdbscanError> {
        validate_params(dim, &params)?;
        let m = params.m.max(1);
        Ok(Self {
            tree: bubble_tree::BubbleTree::new_with_compression(dim, params.compression_rate, m),
            params,
            points: Vec::new(),
            dim,
        })
    }

    pub fn add(&mut self, vectors: &[Vec<f64>]) -> Result<Vec<usize>, HdbscanError> {
        let mut ids = Vec::with_capacity(vectors.len());
        for vector in vectors {
            if vector.len() != self.dim {
                return Err(HdbscanError::InvalidDimension {
                    expected: self.dim,
                    actual: vector.len(),
                });
            }
            let id = self.points.len();
            self.points.push(Some(PointEntry {
                id,
                vector: vector.clone(),
            }));
            self.tree.insert(vector);
            ids.push(id);
        }
        Ok(ids)
    }

    pub fn remove(&mut self, ids: &[usize]) -> Result<(), HdbscanError> {
        for &id in ids {
            if id >= self.points.len() {
                return Err(HdbscanError::PointNotFound(id));
            }
            if let Some(entry) = self.points[id].take() {
                self.tree.delete(&entry.vector);
            }
        }
        Ok(())
    }

    pub fn cluster(&self) -> Result<ClusterResult, HdbscanError> {
        let leaves = self.tree.extract_leaves();
        let bubbles: Vec<DataBubble> = leaves.iter().map(DataBubble::from_cf).collect();

        if bubbles.is_empty() {
            return Err(HdbscanError::NoPoints);
        }

        let hdbscan = Hdbscan::new(self.params.clone());
        hdbscan.cluster(&bubbles)
    }

    pub fn num_bubbles(&self) -> usize {
        self.tree.num_leaves()
    }

    pub fn num_points(&self) -> usize {
        self.tree.total_points()
    }
}

fn validate_params(dim: usize, params: &HdbscanParams) -> Result<(), HdbscanError> {
    if dim == 0 {
        return Err(HdbscanError::InvalidParameter(
            "dim must be greater than 0".to_string(),
        ));
    }
    if params.min_pts == 0 {
        return Err(HdbscanError::InvalidParameter(
            "min_pts must be greater than or equal to 1".to_string(),
        ));
    }
    if params.min_cluster_size < 2 {
        return Err(HdbscanError::InvalidParameter(
            "min_cluster_size must be greater than or equal to 2".to_string(),
        ));
    }
    if !(params.compression_rate > 0.0 && params.compression_rate <= 1.0) {
        return Err(HdbscanError::InvalidParameter(
            "compression_rate must be in the range (0.0, 1.0]".to_string(),
        ));
    }
    if params.m == 0 {
        return Err(HdbscanError::InvalidParameter(
            "m must be greater than or equal to 1".to_string(),
        ));
    }
    if let Some(bit_width) = params.turbovec_bit_width {
        if !(2..=4).contains(&bit_width) {
            return Err(HdbscanError::InvalidParameter(
                "turbovec_bit_width must be 2, 3, or 4".to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_hdbscan() {
        let params = HdbscanParams {
            min_pts: 2,
            min_cluster_size: 2,
            compression_rate: 0.5, // 2 leaves
            m: 1,
            ..Default::default()
        };
        let mut index = HdbscanIncremental::new(2, params);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let ids = index.add(&vectors).unwrap();
        assert_eq!(ids.len(), 4);
        assert_eq!(index.num_points(), 4);

        let result = index.cluster().unwrap();
        // Just verify we get a result without errors
        assert_eq!(result.labels.len(), index.num_bubbles());
    }

    #[test]
    fn test_incremental_hdbscan_add_remove() {
        let params = HdbscanParams {
            min_pts: 2,
            min_cluster_size: 2,
            compression_rate: 0.5, // 2 leaves
            m: 1,
            ..Default::default()
        };
        let mut index = HdbscanIncremental::new(2, params);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let ids = index.add(&vectors).unwrap();
        assert_eq!(index.num_points(), 4);

        index.remove(&[ids[0]]).unwrap();
        assert_eq!(index.num_points(), 3);
    }

    #[test]
    fn test_try_new_rejects_invalid_parameters() {
        assert!(matches!(
            HdbscanIncremental::try_new(0, HdbscanParams::default()),
            Err(HdbscanError::InvalidParameter(_))
        ));

        let mut params = HdbscanParams::default();
        params.compression_rate = 0.0;
        assert!(matches!(
            HdbscanIncremental::try_new(2, params),
            Err(HdbscanError::InvalidParameter(_))
        ));

        let mut params = HdbscanParams::default();
        params.min_cluster_size = 1;
        assert!(matches!(
            HdbscanIncremental::try_new(2, params),
            Err(HdbscanError::InvalidParameter(_))
        ));
    }
}
