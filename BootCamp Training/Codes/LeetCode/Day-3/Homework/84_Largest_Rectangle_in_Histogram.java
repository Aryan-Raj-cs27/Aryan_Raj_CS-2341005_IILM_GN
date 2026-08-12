import java.util.Stack;

class Solution {
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        int n = heights.length;
        
        // Loop through the array, plus one extra iteration for a virtual height of 0
        for (int i = 0; i <= n; i++) {
            // Assign 0 to the current height if we are past the end of the array
            int currentHeight = (i == n) ? 0 : heights[i];
            
            // While the stack is not empty and the current bar is shorter than the top of the stack
            while (!stack.isEmpty() && currentHeight < heights[stack.peek()]) {
                // The height of the rectangle is the bar we just popped
                int h = heights[stack.pop()];
                
                // The width depends on whether the stack becomes empty after popping
                int width = stack.isEmpty() ? i : (i - stack.peek() - 1);
                
                // Update the maximum area
                maxArea = Math.max(maxArea, h * width);
            }
            
            // Push the current index onto the stack
            stack.push(i);
        }
        
        return maxArea;
    }
}