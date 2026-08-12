class Solution {
    public void gameOfLife(int[][] board) {
        int m = board.length;
        int n = board[0].length;
        
        // 8 directions: top-left, top, top-right, left, right, bottom-left, bottom, bottom-right
        int[][] dirs = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int liveNeighbors = 0;
                
                // Count live neighbors
                for (int[] dir : dirs) {
                    int r = i + dir[0];
                    int c = j + dir[1];
                    
                    // Check bounds and if the neighbor is currently alive (LSB is 1)
                    if (r >= 0 && r < m && c >= 0 && c < n && (board[r][c] & 1) == 1) {
                        liveNeighbors++;
                    }
                }
                
                // Apply rules
                if ((board[i][j] & 1) == 1) {
                    // Rule 2: Live cell with 2 or 3 live neighbors lives on
                    if (liveNeighbors == 2 || liveNeighbors == 3) {
                        board[i][j] |= 2; // Set 2nd bit to 1
                    }
                    // Rules 1 & 3: Otherwise it dies (we do nothing, 2nd bit remains 0)
                } else {
                    // Rule 4: Dead cell with exactly 3 live neighbors becomes a live cell
                    if (liveNeighbors == 3) {
                        board[i][j] |= 2; // Set 2nd bit to 1
                    }
                }
            }
        }
        
        // Shift right to throw away the old state and keep only the new state
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] >>= 1;
            }
        }
    }
}