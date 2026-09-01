/**
 * LeetCode 239: Sliding Window Maximum
 * 
 * You are given an array of integers nums, there is a sliding window of size k 
 * which is moving from the very left of the array to the very right.
 * 
 * You can only see the k numbers in the window. Each time the sliding window moves right by one position.
 * Return an array of the maximum sliding window.
 */

import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;

class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return new int[0];
        }
        
        int n = nums.length;
        int[] result = new int[n - k + 1];
        Deque<Integer> deque = new LinkedList<>();
        
        for (int i = 0; i < n; i++) {
            // Remove elements outside the window
            while (!deque.isEmpty() && deque.peekFirst() < i - k + 1) {
                deque.pollFirst();
            }
            
            // Remove elements smaller than current element from the back
            while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
                deque.pollLast();
            }
            
            deque.addLast(i);
            
            // Store the maximum when window is full
            if (i >= k - 1) {
                result[i - k + 1] = nums[deque.peekFirst()];
            }
        }
        
        return result;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {1, 3, -1, -3, 5, 3, 6, 7};
        int k1 = 3;
        System.out.println(Arrays.toString(sol.maxSlidingWindow(nums1, k1)));
        // [3, 3, 5, 5, 6, 7]
        
        int[] nums2 = {1};
        int k2 = 1;
        System.out.println(Arrays.toString(sol.maxSlidingWindow(nums2, k2)));
        // [1]
        
        int[] nums3 = {1, 3, 1, 2, 0, 5};
        int k3 = 3;
        System.out.println(Arrays.toString(sol.maxSlidingWindow(nums3, k3)));
        // [3, 3, 2, 5]
    }
}
