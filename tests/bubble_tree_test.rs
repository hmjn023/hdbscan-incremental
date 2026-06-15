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
