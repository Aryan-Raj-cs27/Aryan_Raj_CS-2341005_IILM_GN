/**
 * LeetCode 17: Letter Combinations of a Phone Number
 * 
 * Given a string containing digits 2-9 inclusive, return all possible letter combinations 
 * that the number could represent. Return the answer in any order.
 * 
 * A mapping of digits to letters (just like on the telephone buttons) is given below.
 * 2: abc, 3: def, 4: ghi, 5: jkl, 6: mno, 7: pqrs, 8: tuv, 9: wxyz
 */

import java.util.ArrayList;
import java.util.List;

class Solution {
    private static final String[] LETTERS = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        
        if (digits == null || digits.length() == 0) {
            return result;
        }
        
        backtrack(result, digits, 0, new StringBuilder());
        return result;
    }
    
    private void backtrack(List<String> result, String digits, int index, StringBuilder current) {
        if (index == digits.length()) {
            result.add(current.toString());
            return;
        }
        
        int digit = digits.charAt(index) - '0';
        String letters = LETTERS[digit];
        
        for (char c : letters.toCharArray()) {
            current.append(c);
            backtrack(result, digits, index + 1, current);
            current.deleteCharAt(current.length() - 1);
        }
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.letterCombinations("23"));   // ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
        System.out.println(sol.letterCombinations(""));     // []
        System.out.println(sol.letterCombinations("2"));    // ["a", "b", "c"]
        System.out.println(sol.letterCombinations("234"));  // 27 combinations
    }
}
