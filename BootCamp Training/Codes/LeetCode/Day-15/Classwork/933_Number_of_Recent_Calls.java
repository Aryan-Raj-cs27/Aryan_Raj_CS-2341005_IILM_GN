/**
 * LeetCode 933: Number of Recent Calls
 * 
 * You have a RecentCounter class that counts the number of recent requests 
 * within a certain time frame.
 * 
 * Implement the RecentCounter class:
 * - RecentCounter() Initializes the counter with zero recent requests.
 * - int ping(int t) Adds a new request at time t, where t represents some time 
 *   in milliseconds, and returns the number of requests that have happened in 
 *   the past 3000 milliseconds (inclusive).
 * 
 * Requests are guaranteed to come in chronological order (each new request is called 
 * with a strictly larger t than the previous request).
 */

import java.util.LinkedList;
import java.util.Queue;

class RecentCounter {
    private Queue<Integer> requests;

    public RecentCounter() {
        requests = new LinkedList<>();
    }
    
    public int ping(int t) {
        requests.offer(t);
        
        // Remove requests older than t - 3000
        while (!requests.isEmpty() && requests.peek() < t - 3000) {
            requests.poll();
        }
        
        return requests.size();
    }

    public static void main(String[] args) {
        RecentCounter counter = new RecentCounter();
        
        System.out.println(counter.ping(1));     // 1
        System.out.println(counter.ping(100));   // 2
        System.out.println(counter.ping(3001));  // 3
        System.out.println(counter.ping(3002));  // 3
    }
}
