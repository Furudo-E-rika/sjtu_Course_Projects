def louvain(G):
    # initialize each node to be in its own community
    communities = {n: i for i, n in enumerate(G.nodes())}
    
    # keep track of the modularity at each iteration
    modularity = compute_modularity(G, communities)
    max_modularity = modularity
    
    while True:
        # keep track of whether any moves have been made during this iteration
        moved = False
        
        # iterate over the nodes in a random order
        nodes = list(G.nodes())
        random.shuffle(nodes)
        
        # find the best community for each node
        for node in nodes:
            # get the current community of the node
            current_community = communities[node]
            
            # find the community that yields the highest increase in modularity if the node is moved there
            best_community = current_community
            best_increase = 0.0
            
            neighbors = list(G.neighbors(node))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor == node:
                    continue
                
                # get the current community of the neighbor
                neighbor_community = communities[neighbor]
                
                # compute the increase in modularity if the node is moved to the neighbor's community
                delta_modularity = compute_delta_modularity(G, node, current_community, neighbor_community)
                
                if delta_modularity > best_increase:
                    best_community = neighbor_community
                    best_increase = delta_modularity
            
            # move the node to the best community, if it improves modularity
            if best_community != current_community:
                communities[node] = best_community
                modularity += best_increase
                moved = True
        
        # stop if no moves were made during this iteration
        if not moved:
            break
        
        # update the modularity and the maximum modularity
        if modularity > max_modularity:
            max_modularity = modularity
    
    # assign a unique ID to each community
    community_id = 0
    community_ids = {}
    
    for node, community in communities.items():
        if community not in community_ids:
            community_ids[community] = community_id
            community_id += 1
        
        G.nodes[node]['category'] = community_ids[community]
