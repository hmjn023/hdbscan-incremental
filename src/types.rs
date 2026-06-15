pub type Vector = Vec<f64>;
pub type Label = i32; // -1 = noise

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterSelection {
    Eom,
    Leaf,
}

#[derive(Debug, Clone)]
pub struct HdbscanParams {
    pub min_pts: usize,
    pub min_cluster_size: usize,
    pub cluster_selection_method: ClusterSelection,
    pub compression_rate: f64,
    pub m: usize,
    /// When set, use turbovec approximate k-NN with the given bit width
    /// for core-distance computation when the number of bubbles is large.
    pub turbovec_bit_width: Option<usize>,
}

impl Default for HdbscanParams {
    fn default() -> Self {
        Self {
            min_pts: 100,
            min_cluster_size: 100,
            cluster_selection_method: ClusterSelection::Eom,
            compression_rate: 0.01,
            m: 25,
            turbovec_bit_width: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub labels: Vec<Label>,
    pub probabilities: Vec<f64>,
    pub num_clusters: usize,
    pub stability: Vec<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PointEntry {
    pub id: usize,
    pub vector: Vector,
}

#[derive(Debug, Clone)]
pub struct LinkageRow {
    pub left: usize,
    pub right: usize,
    pub distance: f64,
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct CondensedTreeNode {
    pub parent: usize,
    pub child: usize,
    pub lambda_val: f64,
    pub child_size: usize,
}

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
