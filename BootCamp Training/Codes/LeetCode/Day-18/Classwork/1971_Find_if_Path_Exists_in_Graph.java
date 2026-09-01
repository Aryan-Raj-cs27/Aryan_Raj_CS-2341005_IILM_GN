/**
 * 1971. Find if Path Exists in Graph
 * https://leetcode.com/problems/find-if-path-exists-in-graph/
 * 
 * There is a bi-directional graph with n vertices labeled from 0 to n - 1 (inclusive)
 * and consists of edges. You are given a 2D integer array edges where edges[i] = [ui, vi]
 * denotes a bi-directional edge between vertices ui and vi.
 * 
 * A valid path from a start vertex to an end vertex is a sequence of vertices where each
 * adjacent pair is connected by an edge in the graph.
 * 
 * Given an integer n, a 2D array edges, an integer source and an integer destination,
 * return true if there is a valid path from source to destination, or false otherwise.
 */

class Solution {
    public boolean validPath(int n, int[][] edges, int source, int destination) {
        // Build adjacency list representation of the graph
        java.util.List<Integer>[] graph = new java.util.ArrayList[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new java.util.ArrayList<>();
        }
        
        // Add edges to the graph (bidirectional)
        for (int[] edge : edges) {
            graph[edge[0]].add(edge[1]);
            graph[edge[1]].add(edge[0]);
        }
        
        // Use BFS to find if path exists from source to destination
        return bfs(graph, source, destination, n);
    }
    
    private boolean bfs(java.util.List<Integer>[] graph, int source, int destination, int n) {
        if (source == destination) {
            return true;
        }
        
        boolean[] visited = new boolean[n];
        java.util.Queue<Integer> queue = new java.util.LinkedList<>();
        queue.offer(source);
        visited[source] = true;
        
        while (!queue.isEmpty()) {
            int current = queue.poll();
            
            // Explore all neighbors
            for (int neighbor : graph[current]) {
                if (neighbor == destination) {
                    return true;
                }
                
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.offer(neighbor);
                }
            }
        }
        
        return false;
    }
}

/*
Alternative DFS Solution:

class Solution {
    public boolean validPath(int n, int[][] edges, int source, int destination) {
        // Build adjacency list
        java.util.List<Integer>[] graph = new java.util.ArrayList[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new java.util.ArrayList<>();
        }
        
        for (int[] edge : edges) {
            graph[edge[0]].add(edge[1]);
            graph[edge[1]].add(edge[0]);
        }
        
        // Use DFS
        boolean[] visited = new boolean[n];
        return dfs(graph, source, destination, visited);
    }
    
    private boolean dfs(java.util.List<Integer>[] graph, int current, int destination, boolean[] visited) {
        if (current == destination) {
            return true;
        }
        
        visited[current] = true;
        
        for (int neighbor : graph[current]) {
            if (!visited[neighbor]) {
                if (dfs(graph, neighbor, destination, visited)) {
                    return true;
                }
            }
        }
        
        return false;
    }
}

Time Complexity: O(n + edges) where n is the number of vertices
                 We visit each vertex at most once and check each edge
Space Complexity: O(n + edges) for the adjacency list and visited array

Approach:
- Build adjacency list representation of the bidirectional graph
- Use BFS or DFS to find if there's a path from source to destination
- Track visited vertices to avoid cycles
- Return true if destination is reached, false otherwise

Example:
Input: n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2
Output: true
Explanation: There are two paths from vertex 0 to vertex 2:
- 0 -> 1 -> 2
- 0 -> 2
*/
