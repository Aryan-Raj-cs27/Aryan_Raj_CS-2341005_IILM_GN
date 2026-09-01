import java.util.Stack;

class Solution {
    public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> stack = new Stack<>();
        for (int a : asteroids) {
            boolean alive = true;
            while (alive && !stack.isEmpty() && stack.peek() > 0 && a < 0) {
                int top = stack.peek();
                if (top < -a) {
                    stack.pop();
                } else if (top == -a) {
                    stack.pop();
                    alive = false;
                } else {
                    alive = false;
                }
            }
            if (alive) stack.push(a);
        }
        int[] result = new int[stack.size()];
        for (int i = result.length - 1; i >= 0; i--) {
            result[i] = stack.pop();
        }
        return result;
    }
}