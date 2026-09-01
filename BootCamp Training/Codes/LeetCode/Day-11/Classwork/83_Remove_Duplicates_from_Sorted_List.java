/**
 * 83. Remove Duplicates from Sorted List
 * https://leetcode.com/problems/remove-duplicates-from-sorted-list/
 * 
 * Given the head of a sorted linked list, delete all duplicates such that
 * each element appears only once. Return the linked list sorted as well.
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
    public ListNode deleteDuplicates(ListNode head) {
        // Edge case: empty list or single node
        if (head == null || head.next == null) {
            return head;
        }
        
        ListNode current = head;
        
        // Traverse the list
        while (current != null && current.next != null) {
            // If current node has same value as next node
            if (current.val == current.next.val) {
                // Skip the next node (which is a duplicate)
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
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        head.next = deleteDuplicates(head.next);
        
        if (head.val == head.next.val) {
            head = head.next;
        }
        
        return head;
    }
}

Time Complexity: O(n) where n is the number of nodes in the list
                 We visit each node exactly once
Space Complexity: O(1) for iterative solution (O(n) for recursive due to call stack)

Approach:
- Traverse the sorted list from the beginning
- Compare each node with its next node
- If values are equal, skip the duplicate node by updating the next pointer
- If values are different, move to the next node
- Continue until reaching the end of the list

Example:
Input: head = [1,1,2]
Output: [1,2]

Input: head = [1,1,2,2,3]
Output: [1,2,3]

Key Point:
- The list is sorted, so all duplicates are adjacent
- We can remove duplicates in a single pass
*/
