/**
 * LeetCode 37: Sudoku Solver
 * 
 * Write a program to solve a Sudoku puzzle by filling the empty cells.
 * 
 * A sudoku solution must satisfy all of the following rules:
 * 1. Each row must contain the digits 1-9 without repetition.
 * 2. Each column must contain the digits 1-9 without repetition.
 * 3. Each of the nine 3x3 sub-boxes of the grid must contain the digits 1-9 without repetition.
 * 
 * The '.' character indicates empty cells.
 */

class Solution {
    public void solveSudoku(char[][] board) {
        backtrack(board);
    }
    
    private boolean backtrack(char[][] board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    for (char c = '1'; c <= '9'; c++) {
                        if (isValid(board, i, j, c)) {
                            board[i][j] = c;
                            
                            if (backtrack(board)) {
                                return true;
                            }
                            
                            board[i][j] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }
    
    private boolean isValid(char[][] board, int row, int col, char c) {
        // Check row
        for (int j = 0; j < 9; j++) {
            if (board[row][j] == c) {
                return false;
            }
        }
        
        // Check column
        for (int i = 0; i < 9; i++) {
            if (board[i][col] == c) {
                return false;
            }
        }
        
        // Check 3x3 box
        int boxRow = (row / 3) * 3;
        int boxCol = (col / 3) * 3;
        for (int i = boxRow; i < boxRow + 3; i++) {
            for (int j = boxCol; j < boxCol + 3; j++) {
                if (board[i][j] == c) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    private void printBoard(char[][] board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        char[][] board = {
            {'5', '3', '.', '.', '7', '.', '.', '.', '.'},
            {'6', '.', '.', '1', '9', '5', '.', '.', '.'},
            {'.', '9', '8', '.', '.', '.', '.', '6', '.'},
            {'8', '.', '.', '.', '6', '.', '.', '.', '3'},
            {'4', '.', '.', '8', '.', '3', '.', '.', '1'},
            {'7', '.', '.', '.', '2', '.', '.', '.', '6'},
            {'.', '6', '.', '.', '.', '.', '2', '8', '.'},
            {'.', '.', '.', '4', '1', '9', '.', '.', '5'},
            {'.', '.', '.', '.', '8', '.', '.', '7', '9'}
        };
        
        System.out.println("Before solving:");
        sol.printBoard(board);
        
        sol.solveSudoku(board);
        
        System.out.println("\nAfter solving:");
        sol.printBoard(board);
    }
}
