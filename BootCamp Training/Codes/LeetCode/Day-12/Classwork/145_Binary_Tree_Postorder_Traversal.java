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
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        postorder(root, result);
        return result;
    }
    
    private void postorder(TreeNode node, List<Integer> result) {
        if (node == null) {
            return;
        }
        
        // Postorder traversal is: Left -> Right -> Root
        postorder(node.left, result);   // Traverse left subtree
        postorder(node.right, result);  // Traverse right subtree
        result.add(node.val);           // Visit the root
    }
}