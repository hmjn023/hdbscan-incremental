use crate::cf::ClusteringFeature;
use crate::distance;

#[derive(Debug, Clone)]
pub struct DataBubble {
    pub rep: Vec<f64>,
    pub n: usize,
    pub extent: f64,
    pub nn_dist_cache: Vec<f64>,
}

impl DataBubble {
    pub fn from_cf(cf: &ClusteringFeature) -> Self {
        let dim = cf.dim();
        let n = cf.n;
        let rep = cf.centroid();

        let extent = if n <= 1 {
            0.0
        } else {
            let numerator = 2.0 * n as f64 * cf.ss - 2.0 * dot_sum_sq(&cf.ls);
            let denominator = n as f64 * (n as f64 - 1.0);
            if denominator > 0.0 {
                (numerator / denominator).sqrt()
            } else {
                0.0
            }
        };

        let max_k = n.min(1024);
        let mut nn_dist_cache = Vec::with_capacity(max_k);
        for k in 1..=max_k {
            nn_dist_cache.push(Self::compute_nn_dist(k, n, dim, extent));
        }

        Self {
            rep,
            n,
            extent,
            nn_dist_cache,
        }
    }

    fn compute_nn_dist(k: usize, n: usize, dim: usize, extent: f64) -> f64 {
        if n == 0 || extent == 0.0 {
            return 0.0;
        }
        let ratio = k as f64 / n as f64;
        let d = dim as f64;
        ratio.powf(1.0 / d) * extent
    }

    pub fn nn_dist(&self, k: usize) -> f64 {
        if k == 0 {
            return 0.0;
        }
        if k <= self.nn_dist_cache.len() {
            self.nn_dist_cache[k - 1]
        } else {
            let dim = self.rep.len();
            Self::compute_nn_dist(k, self.n, dim, self.extent)
        }
    }

    pub fn core_distance(&self, other: &Self, k: usize) -> f64 {
        let d = distance::cosine_distance(&self.rep, &other.rep);
        d + other.nn_dist(k)
    }

    pub fn mutual_reachability(&self, other: &Self, k: usize) -> f64 {
        let cd_self = self.core_distance(other, k);
        let cd_other = other.core_distance(self, k);
        let d = distance::cosine_distance(&self.rep, &other.rep);
        cd_self.max(cd_other).max(d)
    }

    pub fn quality_index(&self, total_n: usize) -> f64 {
        if total_n == 0 {
            return 0.0;
        }
        self.n as f64 / total_n as f64
    }
}

fn dot_sum_sq(ls: &[f64]) -> f64 {
    ls.iter().map(|x| x * x).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bubble_from_cf_single_point() {
        let cf = ClusteringFeature::from_point(&[1.0, 2.0, 3.0]);
        let bubble = DataBubble::from_cf(&cf);
        assert_eq!(bubble.rep, vec![1.0, 2.0, 3.0]);
        assert_eq!(bubble.n, 1);
        assert_eq!(bubble.extent, 0.0);
    }

    #[test]
    fn test_bubble_from_cf_multiple_points() {
        let cf1 = ClusteringFeature::from_point(&[1.0, 0.0]);
        let cf2 = ClusteringFeature::from_point(&[0.0, 1.0]);
        let cf = cf1.merge(&cf2);
        let bubble = DataBubble::from_cf(&cf);
        assert_eq!(bubble.n, 2);
        assert!(bubble.extent > 0.0);
    }

    #[test]
    fn test_nn_dist() {
        let cf = ClusteringFeature::from_point(&[1.0, 2.0]);
        let bubble = DataBubble::from_cf(&cf);
        assert_eq!(bubble.nn_dist(1), 0.0);
    }

    #[test]
    fn test_quality_index() {
        let cf = ClusteringFeature::from_point(&[1.0]);
        let bubble = DataBubble::from_cf(&cf);
        assert!((bubble.quality_index(100) - 0.01).abs() < 1e-10);
    }
}
