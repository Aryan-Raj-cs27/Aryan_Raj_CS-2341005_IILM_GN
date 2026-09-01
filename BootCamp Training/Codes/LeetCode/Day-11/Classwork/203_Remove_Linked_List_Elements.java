/**
 * 203. Remove Linked List Elements
 * https://leetcode.com/problems/remove-linked-list-elements/
 * 
 * Given the head of a linked list and an integer val, remove all the nodes
 * of the linked list that has Node.val == val, and return the new head.
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
    public ListNode removeElements(ListNode head, int val) {
        // Handle the case where head nodes need to be removed
        while (head != null && head.val == val) {
            head = head.next;
        }
        
        // If head is null, the entire list was removed
        if (head == null) {
            return null;
        }
        
        ListNode current = head;
        
        // Traverse and remove nodes with value == val
        while (current != null && current.next != null) {
            if (current.next.val == val) {
                // Skip the node with value == val
                current.next = current.next.next;
            } else {
                // Move to next node
                current = current.next;
            }
        }
        
        return head;
    }
}

/*
Alternative Recursive Solution:

class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        
        head.next = removeElements(head.next, val);
        
        if (head.val == val) {
            return head.next;
        } else {
            return head;
        }
    }
}

Alternative Solution using Dummy Node:

class Solution {
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode current = dummy;
        
        while (current.next != null) {
            if (current.next.val == val) {
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }
        
        return dummy.next;
    }
}

Time Complexity: O(n) where n is the number of nodes in the list
                 We visit each node exactly once
Space Complexity: O(1) for iterative solution (O(n) for recursive due to call stack)

Approach:
- First, handle the case where head nodes need to be removed
- Then, traverse the rest of the list and remove nodes with value == val
- Keep track of the current node and check if the next node should be removed
- If yes, skip it by updating the next pointer

Example:
Input: head = [1,2,6,3,4,5,6], val = 6
Output: [1,2,3,4,5]

Input: head = [7,7,7,7], val = 7
Output: []

Key Points:
- The head itself might need to be removed
- Need to handle the case where the entire list is removed
- Use a dummy node approach for cleaner code
*/
