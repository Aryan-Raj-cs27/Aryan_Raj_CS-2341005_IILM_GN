import java.util.Stack;

class Solution 
{
    public int calPoints(String[] operations) 
    {
        Stack<Integer> stack = new Stack<>();
        for(String op: operations)
        {
            switch(op)
            {
                case "+":
                    int top = stack.pop();
                    int newTop = top+stack.peek();
                    stack.push(top);
                    stack.push(newTop);
                    break;
                case "D":
                    stack.push(stack.peek()*2);
                    break;
                case "C":
                    stack.pop();
                    break;
                default:
                    stack.push(Integer.parseInt(op));
                    break;
            }
        }

        int ts=0;
        for(int s:stack)
        {
            ts+=s;
        }
        return ts;        
    }
}