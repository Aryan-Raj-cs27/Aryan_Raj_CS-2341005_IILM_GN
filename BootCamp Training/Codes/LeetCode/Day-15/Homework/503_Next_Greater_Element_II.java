/**
 * LeetCode 503: Next Greater Element II
 * 
 * Given a circular integer array nums (i.e., the last element's next element is the first element),
 * return an array of the next greater numbers for every element in nums.
 * 
 * The next greater number of a number x is the first greater number to its traversing-order next
 * in the array, which means you could search circularly to find its next greater than itself.
 * 
 * If it doesn't exist, return -1 for this number.
 */

import java.util.Arrays;
import java.util.Stack;

class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        Arrays.fill(result, -1);
        
        Stack<Integer> stack = new Stack<>();
        
        // Traverse array twice to handle circular nature
        for (int i = 0; i < 2 * n; i++) {
            int num = nums[i % n];
            
            // Pop elements from stack if current number is greater
            while (!stack.isEmpty() && nums[stack.peek()] < num) {
                result[stack.pop()] = num;
            }
            
            // Only push index for first traversal
            if (i < n) {
                stack.push(i);
            }
        }
        
        return result;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {1, 2, 1};
        System.out.println(Arrays.toString(sol.nextGreaterElements(nums1)));
        // [2, -1, 2]
        
        int[] nums2 = {1, 2, 3, 4, 3};
        System.out.println(Arrays.toString(sol.nextGreaterElements(nums2)));
        // [2, 3, 4, -1, 4]
    }
}
