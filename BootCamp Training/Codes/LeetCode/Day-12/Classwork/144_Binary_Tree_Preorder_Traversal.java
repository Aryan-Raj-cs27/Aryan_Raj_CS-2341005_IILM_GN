import java.util.ArrayList;
import java.util.List;
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
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        preorder(root, result);
        return result;
    }
    
    private void preorder(TreeNode node, List<Integer> result) {
        if (node == null) {
            return;
        }
        
        // Preorder traversal is: Root -> Left -> Right
        result.add(node.val);          // Visit the root
        preorder(node.left, result);   // Traverse left subtree
        preorder(node.right, result);  // Traverse right subtree
    }
}