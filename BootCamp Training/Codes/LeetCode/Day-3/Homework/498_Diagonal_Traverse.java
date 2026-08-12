class Solution {
    public int[] findDiagonalOrder(int[][] mat) {
        if (mat == null || mat.length == 0) {
            return new int[0];
        }
        
        int m = mat.length;
        int n = mat[0].length;
        int[] result = new int[m * n];
        
        int row = 0;
        int col = 0;
        boolean goingUp = true;
        
        for (int i = 0; i < result.length; i++) {
            // Record the current element
            result[i] = mat[row][col];
            
            // Calculate the next cell based on the current direction
            if (goingUp) {
                if (col == n - 1) {
                    // Hit the right edge: go down, change direction
                    row++;
                    goingUp = false;
                } else if (row == 0) {
                    // Hit the top edge: go right, change direction
                    col++;
                    goingUp = false;
                } else {
                    // Normal up-right move
                    row--;
                    col++;
                }
            } else {
                if (row == m - 1) {
                    // Hit the bottom edge: go right, change direction
                    col++;
                    goingUp = true;
                } else if (col == 0) {
                    // Hit the left edge: go down, change direction
                    row++;
                    goingUp = true;
                } else {
                    // Normal down-left move
                    row++;
                    col--;
                }
            }
        }
        
        return result;
    }
}