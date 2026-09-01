
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
