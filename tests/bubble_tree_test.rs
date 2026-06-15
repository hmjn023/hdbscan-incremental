use hdbscan_incremental::bubble_tree::BubbleTree;

#[test]
fn test_leaf_count_after_inserts() {
    let mut tree = BubbleTree::new(2, 2, 1);
    tree.insert(&[0.0, 0.0]);
    assert_eq!(tree.num_leaves(), 1, "after 1 insert");
    tree.insert(&[0.1, 0.1]);
    assert_eq!(tree.num_leaves(), 2, "after 2 inserts");
    tree.insert(&[10.0, 10.0]);
    assert_eq!(tree.num_leaves(), 2, "after 3 inserts");
    tree.insert(&[10.1, 10.1]);
    assert_eq!(tree.num_leaves(), 2, "after 4 inserts");
}

#[test]
fn test_compression_rate_target_leaf_calculation() {
    assert_eq!(BubbleTree::target_leaves_for(4, 0.25), 1);
    assert_eq!(BubbleTree::target_leaves_for(5, 0.25), 2);
    assert_eq!(BubbleTree::target_leaves_for(10_001, 0.01), 101);
}
