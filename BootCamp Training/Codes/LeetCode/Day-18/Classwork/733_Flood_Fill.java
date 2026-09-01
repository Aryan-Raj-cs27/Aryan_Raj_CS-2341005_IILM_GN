/**
 * 733. Flood Fill
 * https://leetcode.com/problems/flood-fill/
 * 
 * An image is represented by an m x n integer grid where each integer represents a pixel value.
 * You are also given three integers sr, sc, and newColor. You should perform a "flood fill"
 * on the image starting from the pixel image[sr][sc].
 * 
 * To perform a "flood fill":
 * 1. Start from the pixel image[sr][sc]
 * 2. Change it to newColor
 * 3. Change any directly adjacent pixels of the same original color to newColor
 * 4. Recursively repeat for any newly changed pixels
 */

class Solution {
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int originalColor = image[sr][sc];
        
        // If the new color is same as original, no need to fill
        if (originalColor == newColor) {
            return image;
        }
        
        // Call DFS to perform flood fill
        dfs(image, sr, sc, originalColor, newColor);
        
        return image;
    }
    
    private void dfs(int[][] image, int row, int col, int originalColor, int newColor) {
        // Base cases: out of bounds or cell is not the original color
        if (row < 0 || row >= image.length || col < 0 || col >= image[0].length 
            || image[row][col] != originalColor) {
            return;
        }
        
        // Change the current cell to new color
        image[row][col] = newColor;
        
        // Recursively fill all 4 adjacent cells
        dfs(image, row + 1, col, originalColor, newColor);  // down
        dfs(image, row - 1, col, originalColor, newColor);  // up
        dfs(image, row, col + 1, originalColor, newColor);  // right
        dfs(image, row, col - 1, originalColor, newColor);  // left
    }
}

/*
Time Complexity: O(m * n) where m is number of rows and n is number of columns
                 In worst case, we visit every cell once
Space Complexity: O(m * n) for the recursion stack in worst case
                  (imagine a large connected region)

Approach:
- Use DFS (Depth-First Search) to traverse connected pixels
- Start from (sr, sc) and explore in all 4 directions
- Change each visited pixel from original color to new color
- Stop when reaching boundaries or pixels that aren't the original color

Example:
Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
- Starting from (1,1) with color 1, all connected 1s become 2
*/
