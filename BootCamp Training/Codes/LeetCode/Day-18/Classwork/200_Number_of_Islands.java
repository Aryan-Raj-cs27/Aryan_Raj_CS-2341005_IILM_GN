/**
 * 200. Number of Islands
 * https://leetcode.com/problems/number-of-islands/
 * 
 * Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water),
 * return the number of islands.
 * 
 * An island is surrounded by water and is formed by connecting adjacent lands horizontally
 * or vertically. You may assume all four edges of the grid are surrounded by water.
 */

class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        
        int rows = grid.length;
        int cols = grid[0].length;
        int islandCount = 0;
        
        // Traverse the entire grid
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // When we find land ('1'), increment island count and explore it
                if (grid[i][j] == '1') {
                    islandCount++;
                    dfs(grid, i, j, rows, cols);
                }
            }
        }
        
        return islandCount;
    }
    
    private void dfs(char[][] grid, int row, int col, int rows, int cols) {
        // Base cases: out of bounds or cell is water
        if (row < 0 || row >= rows || col < 0 || col >= cols || grid[row][col] == '0') {
            return;
        }
        
        // Mark current cell as visited by changing it to '0'
        grid[row][col] = '0';
        
        // Explore all 4 adjacent cells
        dfs(grid, row + 1, col, rows, cols);  // down
        dfs(grid, row - 1, col, rows, cols);  // up
        dfs(grid, row, col + 1, rows, cols);  // right
        dfs(grid, row, col - 1, rows, cols);  // left
    }
}

/*
Alternative BFS Solution:

class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        
        int rows = grid.length;
        int cols = grid[0].length;
        int islandCount = 0;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == '1') {
                    islandCount++;
                    bfs(grid, i, j, rows, cols);
                }
            }
        }
        
        return islandCount;
    }
    
    private void bfs(char[][] grid, int row, int col, int rows, int cols) {
        java.util.Queue<int[]> queue = new java.util.LinkedList<>();
        queue.offer(new int[]{row, col});
        grid[row][col] = '0';
        
        int[][] directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        
        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int r = current[0];
            int c = current[1];
            
            for (int[] dir : directions) {
                int newRow = r + dir[0];
                int newCol = c + dir[1];
                
                if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols 
                    && grid[newRow][newCol] == '1') {
                    grid[newRow][newCol] = '0';
                    queue.offer(new int[]{newRow, newCol});
                }
            }
        }
    }
}

Alternative Union-Find Solution:

class Solution {
    class UnionFind {
        int[] parent;
        int[] rank;
        int components;
        
        public UnionFind(char[][] grid) {
            int rows = grid.length;
            int cols = grid[0].length;
            parent = new int[rows * cols];
            rank = new int[rows * cols];
            components = 0;
            
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (grid[i][j] == '1') {
                        parent[i * cols + j] = i * cols + j;
                        components++;
                    }
                }
            }
        }
        
        public int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }
        
        public void union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            
            if (rootX == rootY) return;
            
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            components--;
        }
        
        public int getComponents() {
            return components;
        }
    }
    
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        
        UnionFind uf = new UnionFind(grid);
        int rows = grid.length;
        int cols = grid[0].length;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == '1') {
                    // Union with right neighbor
                    if (j + 1 < cols && grid[i][j + 1] == '1') {
                        uf.union(i * cols + j, i * cols + j + 1);
                    }
                    // Union with down neighbor
                    if (i + 1 < rows && grid[i + 1][j] == '1') {
                        uf.union(i * cols + j, (i + 1) * cols + j);
                    }
                }
            }
        }
        
        return uf.getComponents();
    }
}

Time Complexity: O(m * n) where m is number of rows and n is number of columns
                 We visit each cell at most once
Space Complexity: O(m * n) for the recursion stack in worst case (DFS)
                  Or O(m * n) for the queue (BFS)

Approach:
- Use DFS/BFS to explore each island
- When we encounter a '1', it's a new island, so increment count
- Then explore all connected land cells in that island and mark them as visited
- Continue until all cells are explored
- Return the total number of islands found

Example:
Input: grid = [["1","1","1","1","0"],
               ["1","1","0","1","0"],
               ["1","1","0","0","0"],
               ["0","0","0","0","0"]]
Output: 1
Explanation: All 1's form one connected island
*/
