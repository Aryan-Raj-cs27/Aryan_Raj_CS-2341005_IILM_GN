class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        
        // If we reach a leaf node, check if its value equals the remaining target sum
        if (root.left == null && root.right == null) {
            return targetSum == root.val;
        }
        
        // Recursively check the left and right subtrees with the updated target sum
        return hasPathSum(root.left, targetSum - root.val) || 
               hasPathSum(root.right, targetSum - root.val);
    }
}