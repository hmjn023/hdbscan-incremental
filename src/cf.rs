use crate::distance;
use crate::types::Vector;

#[derive(Debug, Clone)]
pub struct ClusteringFeature {
    pub ls: Vec<f64>,
    pub ss: f64,
    pub n: usize,
}

impl ClusteringFeature {
    pub fn from_point(p: &[f64]) -> Self {
        let ss = p.iter().map(|x| x * x).sum();
        Self {
            ls: p.to_vec(),
            ss,
            n: 1,
        }
    }

    pub fn merge(&self, other: &Self) -> Self {
        debug_assert_eq!(self.ls.len(), other.ls.len());
        let ls = self
            .ls
            .iter()
            .zip(other.ls.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self {
            ls,
            ss: self.ss + other.ss,
            n: self.n + other.n,
        }
    }

    pub fn centroid(&self) -> Vector {
        self.ls.iter().map(|x| x / self.n as f64).collect()
    }

    pub fn add_point(&mut self, p: &[f64]) {
        debug_assert_eq!(self.ls.len(), p.len());
        for (ls, x) in self.ls.iter_mut().zip(p.iter()) {
            *ls += x;
        }
        self.ss += p.iter().map(|x| x * x).sum::<f64>();
        self.n += 1;
    }

    pub fn remove_point(&mut self, p: &[f64]) {
        debug_assert_eq!(self.ls.len(), p.len());
        debug_assert!(self.n > 0);
        for (ls, x) in self.ls.iter_mut().zip(p.iter()) {
            *ls -= x;
        }
        self.ss -= p.iter().map(|x| x * x).sum::<f64>();
        self.n -= 1;
    }

    pub fn dim(&self) -> usize {
        self.ls.len()
    }

    pub fn distance_to_point(&self, p: &[f64]) -> f64 {
        let centroid = self.centroid();
        distance::cosine_distance(&centroid, p)
    }

    pub fn distance_to_cf(&self, other: &Self) -> f64 {
        let c1 = self.centroid();
        let c2 = other.centroid();
        distance::cosine_distance(&c1, &c2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cf_from_point() {
        let p = vec![1.0, 2.0, 3.0];
        let cf = ClusteringFeature::from_point(&p);
        assert_eq!(cf.ls, p);
        assert_eq!(cf.ss, 14.0);
        assert_eq!(cf.n, 1);
    }

    #[test]
    fn test_cf_merge() {
        let cf1 = ClusteringFeature::from_point(&[1.0, 2.0]);
        let cf2 = ClusteringFeature::from_point(&[3.0, 4.0]);
        let merged = cf1.merge(&cf2);
        assert_eq!(merged.ls, vec![4.0, 6.0]);
        assert_eq!(merged.ss, 30.0);
        assert_eq!(merged.n, 2);
    }

    #[test]
    fn test_cf_centroid() {
        let cf = ClusteringFeature::from_point(&[2.0, 4.0]);
        let centroid = cf.centroid();
        assert_eq!(centroid, vec![2.0, 4.0]);

        let cf2 = ClusteringFeature::from_point(&[4.0, 8.0]);
        let merged = cf.merge(&cf2);
        let centroid = merged.centroid();
        assert_eq!(centroid, vec![3.0, 6.0]);
    }

    #[test]
    fn test_cf_add_remove() {
        let mut cf = ClusteringFeature::from_point(&[1.0, 2.0]);
        cf.add_point(&[3.0, 4.0]);
        assert_eq!(cf.n, 2);
        assert_eq!(cf.ls, vec![4.0, 6.0]);

        cf.remove_point(&[1.0, 2.0]);
        assert_eq!(cf.n, 1);
        assert_eq!(cf.ls, vec![3.0, 4.0]);
    }

    #[test]
    fn test_cf_additivity() {
        let cf1 = ClusteringFeature::from_point(&[1.0, 2.0]);
        let cf2 = ClusteringFeature::from_point(&[3.0, 4.0]);

        let merged = cf1.merge(&cf2);

        let mut cf3 = cf1.clone();
        cf3.add_point(&[3.0, 4.0]);

        assert_eq!(merged.ls, cf3.ls);
        assert!((merged.ss - cf3.ss).abs() < 1e-10);
        assert_eq!(merged.n, cf3.n);
    }
}
