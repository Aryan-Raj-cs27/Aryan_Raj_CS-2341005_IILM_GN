/**
 * LeetCode 51: N-Queens
 * 
 * The n-queens puzzle is the problem of placing n queens on an n x n chessboard 
 * such that no two queens attack each other.
 * 
 * Given an integer n, return all distinct solutions to the n-queens puzzle. 
 * You may return the answer in any order.
 * 
 * Each solution contains a distinct board configuration of the n-queens' placement.
 */

import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> result = new ArrayList<>();
        char[][] board = new char[n][n];
        
        // Initialize board with '.'
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = '.';
            }
        }
        
        backtrack(result, board, 0);
        return result;
    }
    
    private void backtrack(List<List<String>> result, char[][] board, int row) {
        if (row == board.length) {
            result.add(convertToList(board));
            return;
        }
        
        for (int col = 0; col < board.length; col++) {
            if (isValid(board, row, col)) {
                board[row][col] = 'Q';
                backtrack(result, board, row + 1);
                board[row][col] = '.';
            }
        }
    }
    
    private boolean isValid(char[][] board, int row, int col) {
        int n = board.length;
        
        // Check column
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        
        // Check upper-left diagonal
        int i = row - 1, j = col - 1;
        while (i >= 0 && j >= 0) {
            if (board[i][j] == 'Q') {
                return false;
            }
            i--;
            j--;
        }
        
        // Check upper-right diagonal
        i = row - 1;
        j = col + 1;
        while (i >= 0 && j < n) {
            if (board[i][j] == 'Q') {
                return false;
            }
            i--;
            j++;
        }
        
        return true;
    }
    
    private List<String> convertToList(char[][] board) {
        List<String> list = new ArrayList<>();
        for (char[] row : board) {
            list.add(new String(row));
        }
        return list;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.solveNQueens(4));
        // Returns 2 solutions for n=4
        
        System.out.println(sol.solveNQueens(1));
        // Returns 1 solution: [["Q"]]
    }
}
