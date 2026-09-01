/**
 * 21. Merge Two Sorted Lists
 * https://leetcode.com/problems/merge-two-sorted-lists/
 * 
 * Merge two sorted linked lists and return it as a new sorted list.
 * The new list should be made by splicing together the nodes of the two lists.
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
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        // Create a dummy node to simplify the merge process
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        // Traverse both lists and compare nodes
        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                current.next = list1;
                list1 = list1.next;
            } else {
                current.next = list2;
                list2 = list2.next;
            }
            current = current.next;
        }
        
        // Attach the remaining nodes
        if (list1 != null) {
            current.next = list1;
        } else {
            current.next = list2;
        }
        
        return dummy.next;
    }
}

/*
Alternative Recursive Solution:

class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        
        if (list1.val <= list2.val) {
            list1.next = mergeTwoLists(list1.next, list2);
            return list1;
        } else {
            list2.next = mergeTwoLists(list1, list2.next);
            return list2;
        }
    }
}

Time Complexity: O(n + m) where n is length of list1 and m is length of list2
                 We visit each node exactly once
Space Complexity: O(1) for iterative solution (O(n + m) for recursive due to call stack)

Approach:
- Use a dummy node to simplify handling the head of the result list
- Compare nodes from both lists and append the smaller one to the result
- Continue until one list is exhausted
- Attach the remaining nodes from the non-empty list

Example:
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4]
*/
