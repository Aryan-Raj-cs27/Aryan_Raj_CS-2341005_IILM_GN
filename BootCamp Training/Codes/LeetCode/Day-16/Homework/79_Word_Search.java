/**
 * LeetCode 79: Word Search
 * 
 * Given an m x n grid of characters board and a string word, 
 * return true if word exists in the grid.
 * 
 * The word can be constructed from letters of sequentially adjacent cells, 
 * where adjacent cells are horizontally or vertically neighboring.
 * The same letter cell may not be used more than once in a word.
 */

class Solution {
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || word == null || word.length() == 0) {
            return false;
        }
        
        int rows = board.length;
        int cols = board[0].length;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j] == word.charAt(0)) {
                    if (dfs(board, word, i, j, 0)) {
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
    
    private boolean dfs(char[][] board, String word, int row, int col, int index) {
        if (index == word.length()) {
            return true;
        }
        
        if (row < 0 || row >= board.length || col < 0 || col >= board[0].length || 
            board[row][col] != word.charAt(index)) {
            return false;
        }
        
        char temp = board[row][col];
        board[row][col] = ' '; // Mark as visited
        
        boolean found = dfs(board, word, row + 1, col, index + 1) ||
                       dfs(board, word, row - 1, col, index + 1) ||
                       dfs(board, word, row, col + 1, index + 1) ||
                       dfs(board, word, row, col - 1, index + 1);
        
        board[row][col] = temp; // Backtrack
        
        return found;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        char[][] board1 = {{'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}};
        System.out.println(sol.exist(board1, "ABCCED"));  // true
        System.out.println(sol.exist(board1, "SEE"));     // true
        System.out.println(sol.exist(board1, "ABCB"));    // false
    }
}
