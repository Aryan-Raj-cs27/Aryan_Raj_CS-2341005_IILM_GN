/**
 * LeetCode 682: Baseball Game
 * 
 * Record of results of a series of games. Each string represents the information of a game.
 * 
 * Strings that only contain the letter 'C' - an integer representing the number of points you obtained in the previous 2 games. 
 * When you encounter a 'D', the current score is 2 times the score of the previous game.
 * When you encounter a '+', the current score is the sum of the previous two games' scores.
 * Each other string represents an integer and records a new score.
 * 
 * Return the sum of all the scores.
 */

import java.util.Stack;

class Solution {
    public int calPoints(String[] operations) {
        Stack<Integer> stack = new Stack<>();
        
        for (String op : operations) {
            if (op.equals("C")) {
                stack.pop();
            } else if (op.equals("D")) {
                stack.push(stack.peek() * 2);
            } else if (op.equals("+")) {
                int top = stack.pop();
                int newTop = stack.peek() + top;
                stack.push(top);
                stack.push(newTop);
            } else {
                stack.push(Integer.parseInt(op));
            }
        }
        
        int sum = 0;
        while (!stack.isEmpty()) {
            sum += stack.pop();
        }
        return sum;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        String[] ops1 = {"5", "2", "C", "D", "+"};
        System.out.println(sol.calPoints(ops1));  // 30
        
        String[] ops2 = {"5", "-2", "4", "C", "D", "9", "+", "+"};
        System.out.println(sol.calPoints(ops2));  // 27
        
        String[] ops3 = {"1"};
        System.out.println(sol.calPoints(ops3));  // 1
    }
}
