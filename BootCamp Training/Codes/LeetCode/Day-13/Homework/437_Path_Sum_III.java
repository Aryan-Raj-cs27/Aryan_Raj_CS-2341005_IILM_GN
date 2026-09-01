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
    public int pathSum(TreeNode root, int targetSum) {
        Map<Long, Integer> prefix = new HashMap<>();
        prefix.put(0L, 1);
        return dfs(root, 0L, targetSum, prefix);
    }

    private int dfs(TreeNode node, long currentSum, int target, Map<Long, Integer> prefix) {
        if (node == null) return 0;

        currentSum += node.val;
        int count = prefix.getOrDefault(currentSum - target, 0);
        prefix.put(currentSum, prefix.getOrDefault(currentSum, 0) + 1);

        count += dfs(node.left, currentSum, target, prefix);
        count += dfs(node.right, currentSum, target, prefix);

        prefix.put(currentSum, prefix.get(currentSum) - 1);
        if (prefix.get(currentSum) == 0) prefix.remove(currentSum);
        return count;
    }

}
