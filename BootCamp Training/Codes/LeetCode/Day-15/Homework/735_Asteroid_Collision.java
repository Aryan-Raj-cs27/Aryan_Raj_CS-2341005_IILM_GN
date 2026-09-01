/**
 * LeetCode 735: Asteroid Collision
 * 
 * We are given an array asteroids of integers representing asteroids in a row.
 * For each asteroid, the absolute value represents its size and the sign represents its direction
 * (positive meaning right, negative meaning left). Each asteroid moves at the same speed.
 * 
 * Find out the state of the asteroids after all collisions (if any occur).
 * 
 * Rules for collision:
 * If two asteroids meet, the smaller one will explode.
 * If both are the same size, both will explode.
 * Two asteroids moving in the same direction will never collide.
 */

import java.util.Arrays;
import java.util.Stack;

class Solution {
    public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> stack = new Stack<>();
        
        for (int asteroid : asteroids) {
            boolean alive = true;
            
            while (alive && asteroid < 0 && !stack.isEmpty() && stack.peek() > 0) {
                int top = stack.peek();
                
                if (top < -asteroid) {
                    // Right moving asteroid explodes
                    stack.pop();
                } else if (top == -asteroid) {
                    // Both explode
                    stack.pop();
                    alive = false;
                } else {
                    // Left moving asteroid explodes
                    alive = false;
                }
            }
            
            if (alive) {
                stack.push(asteroid);
            }
        }
        
        int[] result = new int[stack.size()];
        for (int i = result.length - 1; i >= 0; i--) {
            result[i] = stack.pop();
        }
        
        return result;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] asteroids1 = {5, 10, -5};
        System.out.println(Arrays.toString(sol.asteroidCollision(asteroids1)));
        // [5, 10]
        
        int[] asteroids2 = {8, -8};
        System.out.println(Arrays.toString(sol.asteroidCollision(asteroids2)));
        // []
        
        int[] asteroids3 = {10, 2, -5};
        System.out.println(Arrays.toString(sol.asteroidCollision(asteroids3)));
        // [10]
    }
}
