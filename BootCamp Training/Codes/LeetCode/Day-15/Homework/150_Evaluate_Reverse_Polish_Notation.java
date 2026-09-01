/**
 * LeetCode 150: Evaluate Reverse Polish Notation
 * 
 * Evaluate the value of an arithmetic expression in Reverse Polish Notation.
 * 
 * Valid operators are +, -, *, and /. Each operand may be an integer or another expression.
 * 
 * Note that division between two integers should truncate toward zero.
 * It is guaranteed that the given RPN expression is always valid. That is:
 * - The expression would always evaluate to a result, and
 * - There will not be any division by zero operation.
 */

import java.util.Stack;

class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        
        for (String token : tokens) {
            if (token.equals("+") || token.equals("-") || token.equals("*") || token.equals("/")) {
                int b = stack.pop();
                int a = stack.pop();
                
                int result = 0;
                switch (token) {
                    case "+":
                        result = a + b;
                        break;
                    case "-":
                        result = a - b;
                        break;
                    case "*":
                        result = a * b;
                        break;
                    case "/":
                        result = a / b;
                        break;
                }
                stack.push(result);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        
        return stack.pop();
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        String[] tokens1 = {"2", "1", "+", "3", "*"};
        System.out.println(sol.evalRPN(tokens1));  // 9
        
        String[] tokens2 = {"4", "13", "5", "/", "+"};
        System.out.println(sol.evalRPN(tokens2));  // 6
        
        String[] tokens3 = {"10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"};
        System.out.println(sol.evalRPN(tokens3));  // 22
    }
}
