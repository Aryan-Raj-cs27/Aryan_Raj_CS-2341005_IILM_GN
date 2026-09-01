/**
 * LeetCode 496: Next Greater Element I
 * 
 * The next greater element of some element x in an array is the first greater element 
 * to its right in the same array.
 * 
 * You are given two distinct 0-indexed integer arrays nums1 and nums2, 
 * where nums1 is a subset of nums2.
 * 
 * For each element nums1[i], find the index j such that nums1[i] == nums2[j] 
 * and determine the next greater element of nums2[j] in nums2. 
 * If there is no next greater element, then the answer for this query is -1.
 * 
 * Return an array ans of length nums1.length such that ans[i] is the next greater element.
 */

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

class Solution {
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Stack<Integer> stack = new Stack<>();
        Map<Integer, Integer> map = new HashMap<>();
        
        // Build map of element -> next greater element
        for (int num : nums2) {
            while (!stack.isEmpty() && stack.peek() < num) {
                map.put(stack.pop(), num);
            }
            stack.push(num);
        }
        
        int[] result = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            result[i] = map.getOrDefault(nums1[i], -1);
        }
        
        return result;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {4, 1, 2};
        int[] nums2 = {1, 3, 4, 2};
        System.out.println(Arrays.toString(sol.nextGreaterElement(nums1, nums2))); // [-1, 3, -1]
        
        int[] nums1b = {2, 4};
        int[] nums2b = {1, 2, 3, 4};
        System.out.println(Arrays.toString(sol.nextGreaterElement(nums1b, nums2b))); // [3, -1]
    }
}
