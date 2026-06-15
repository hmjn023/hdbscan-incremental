pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 && norm_b == 0.0 {
        return 0.0;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - dot / (norm_a * norm_b)
}

pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance_same_vector() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &a) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_distance(&a, &b) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance_zero_vectors() {
        let zero = vec![0.0, 0.0];
        let non_zero = vec![1.0, 0.0];

        assert!((cosine_distance(&zero, &zero) - 0.0).abs() < 1e-10);
        assert!((cosine_distance(&zero, &non_zero) - 1.0).abs() < 1e-10);
        assert!((cosine_distance(&non_zero, &zero) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance_same() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((euclidean_distance(&a, &a) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance_simple() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
    }
}
