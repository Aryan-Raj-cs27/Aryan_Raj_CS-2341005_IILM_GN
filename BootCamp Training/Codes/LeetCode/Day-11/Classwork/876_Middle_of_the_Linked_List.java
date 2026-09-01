/**
 * 876. Middle of the Linked List
 * https://leetcode.com/problems/middle-of-the-linked-list/
 * 
 * Given the head of a singly linked list, return the middle node of the linked list.
 * If there are two middle nodes, return the second middle node.
 */

/**
 * Definition for singly-linked list.
 */
class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

class Solution {
    public ListNode middleNode(ListNode head) {
        // Use slow and fast pointer approach
        // slow moves 1 step, fast moves 2 steps
        ListNode slow = head;
        ListNode fast = head;
        
        // When fast reaches the end, slow will be at the middle
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        return slow;
    }
}

/*
Alternative Solution (Two Pass):

class Solution {
    public ListNode middleNode(ListNode head) {
        // First pass: count the total number of nodes
        int count = 0;
        ListNode temp = head;
        while (temp != null) {
            count++;
            temp = temp.next;
        }
        
        // Find the position of the middle node
        int middle = count / 2;
        
        // Second pass: reach the middle node
        temp = head;
        for (int i = 0; i < middle; i++) {
            temp = temp.next;
        }
        
        return temp;
    }
}

Time Complexity: O(n) where n is the number of nodes in the list
                 We traverse the list at most once
Space Complexity: O(1) - only using constant extra space

Approach:
- Use the slow and fast pointer technique (tortoise and hare)
- slow pointer moves one step at a time
- fast pointer moves two steps at a time
- When fast pointer reaches the end, slow pointer will be at the middle
- If there are even number of nodes, we get the second middle node

Why this works:
- Let n be the number of nodes
- Slow moves n/2 steps
- Fast moves 2*(n/2) = n steps when n is even
- Fast moves 2*((n+1)/2) = n+1 steps when n is odd, so it ends at null
- In both cases, slow ends at the middle

Example:
Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node is node 3

Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
Explanation: There are two middle nodes 3 and 4, return the second middle node

Key Points:
- Slow and fast pointer is more efficient than counting first
- Works correctly for both odd and even length lists
- Returns the second middle node for even-length lists
*/
