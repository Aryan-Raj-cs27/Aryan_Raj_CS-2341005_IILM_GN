import java.util.HashMap;
import java.util.Map;

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
    private Map<Integer, Integer> inorderIndex = new HashMap<>();
    private int postIndex;

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        for (int i = 0; i < inorder.length; i++) {
            inorderIndex.put(inorder[i], i);
        }
        postIndex = postorder.length - 1;
        return build(inorder, 0, inorder.length - 1, postorder);
    }

    private TreeNode build(int[] inorder, int left, int right, int[] postorder) {
        if (left > right) return null;

        int rootVal = postorder[postIndex--];
        TreeNode root = new TreeNode(rootVal);
        int mid = inorderIndex.get(rootVal);

        root.right = build(inorder, mid + 1, right, postorder);
        root.left = build(inorder, left, mid - 1, postorder);
        return root;
    }

}
