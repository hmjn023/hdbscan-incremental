use hdbscan_incremental::cf::ClusteringFeature;
use hdbscan_incremental::data_bubble::DataBubble;
use hdbscan_incremental::hdbscan::Hdbscan;
use hdbscan_incremental::HdbscanParams;

fn make_bubble(points: &[Vec<f64>]) -> DataBubble {
    let mut cf = ClusteringFeature::from_point(&points[0]);
    for p in &points[1..] {
        cf.add_point(p);
    }
    DataBubble::from_cf(&cf)
}

#[test]
fn test_hdbscan_detects_two_bubble_clusters() {
    let params = HdbscanParams {
        min_pts: 2,
        min_cluster_size: 2,
        compression_rate: 0.25,
        m: 1,
        ..Default::default()
    };

    let bubbles = vec![
        make_bubble(&[vec![0.0, 0.0], vec![0.1, 0.1]]),
        make_bubble(&[vec![0.2, 0.2], vec![0.3, 0.3]]),
        make_bubble(&[vec![10.0, 10.0], vec![10.1, 10.1]]),
        make_bubble(&[vec![10.2, 10.2], vec![10.3, 10.3]]),
    ];

    let hdbscan = Hdbscan::new(params);
    let result = hdbscan.cluster(&bubbles).unwrap();

    assert!(
        result.num_clusters >= 2,
        "expected at least two clusters, got {:?}",
        result.labels
    );
}
