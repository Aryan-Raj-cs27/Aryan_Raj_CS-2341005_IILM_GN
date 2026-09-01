/**
 * LeetCode 739: Daily Temperatures
 * 
 * Given an array of integers temperatures representing the daily temperatures, 
 * return an array answer such that answer[i] is the number of days you have to wait 
 * after the ith day to get a warmer temperature. 
 * 
 * If there is no future day for which this is possible, keep answer[i] == 0 instead.
 */

import java.util.Arrays;
import java.util.Stack;

class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] answer = new int[n];
        Stack<Integer> stack = new Stack<>();
        
        for (int i = 0; i < n; i++) {
            // Pop stack while current temp is greater than temperature at stack top
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int prevIndex = stack.pop();
                answer[prevIndex] = i - prevIndex;
            }
            stack.push(i);
        }
        
        return answer;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] temps1 = {73, 74, 75, 71, 69, 72, 76, 73};
        System.out.println(Arrays.toString(sol.dailyTemperatures(temps1)));
        // [1, 1, 4, 2, 1, 1, 0, 0]
        
        int[] temps2 = {30, 40, 50, 60};
        System.out.println(Arrays.toString(sol.dailyTemperatures(temps2)));
        // [1, 1, 1, 0]
        
        int[] temps3 = {30, 60, 90};
        System.out.println(Arrays.toString(sol.dailyTemperatures(temps3)));
        // [1, 1, 0]
    }
}
