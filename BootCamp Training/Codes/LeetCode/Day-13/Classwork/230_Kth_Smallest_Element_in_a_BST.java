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
    private int count = 0;
    private int answer = -1;

    public int kthSmallest(TreeNode root, int k) {
        inorder(root, k);
        return answer;
    }

    private void inorder(TreeNode node, int k) {
        if (node == null || count >= k) return;
        inorder(node.left, k);
        count++;
        if (count == k) {
            answer = node.val;
            return;
        }
        inorder(node.right, k);
    }
}
