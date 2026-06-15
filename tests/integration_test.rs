use hdbscan_incremental::{HdbscanIncremental, HdbscanParams};

fn small_params() -> HdbscanParams {
    HdbscanParams {
        min_pts: 2,
        min_cluster_size: 1,
        compression_rate: 0.25,
        m: 1,
        ..Default::default()
    }
}

#[test]
fn test_detects_two_clusters() {
    let mut index = HdbscanIncremental::new(2, small_params());

    let vectors = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.2, 0.2],
        vec![0.3, 0.3],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
        vec![10.2, 10.2],
        vec![10.3, 10.3],
    ];

    index.add(&vectors).unwrap();
    let result = index.cluster().unwrap();

    assert!(
        result.num_clusters >= 2,
        "expected at least two clusters, got {:?}",
        result.labels
    );
}

#[test]
fn test_add_remove_preserves_state() {
    let mut index = HdbscanIncremental::new(2, small_params());

    let vectors = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
    ];

    let ids = index.add(&vectors).unwrap();
    assert_eq!(index.num_points(), 4);
    assert!(index.num_bubbles() > 0);

    index.remove(&[ids[0]]).unwrap();
    assert_eq!(index.num_points(), 3);
}
