#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import pandas as pd
import netrd
from scipy.stats import rankdata
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigvalsh
from time import sleep
from tqdm.notebook import tqdm
import seaborn as sb
import matplotlib.pyplot as plt
import scipy
import random
import time


# # Similarity Measures

# In[5]:


sim_dict = dict()


# ## 3.1.1 Jaccard index

# In[6]:


from iteration_utilities import first

def jaccard(G1,G2):   
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    return len(E1&E2)/len(E1|E2)

def jaccard_weighted(G1,G2, attr="weight"):   
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    if (len(E1)>0 and attr in G1[first(E1)[0]][first(E1)[1]]) or (len(E2)>0 and attr in G2[first(E2)[0]][first(E2)[1]]):
        E_overlap = E1&E2
        n1, n2 = 0, 0
        for edge in E_overlap:
            n1+=min(G1[edge[0]][edge[1]][attr], G2[edge[0]][edge[1]][attr])
            n2+=max(G1[edge[0]][edge[1]][attr], G2[edge[0]][edge[1]][attr])
        E1u, E2u = E1 - E_overlap, E2 - E_overlap
        for edge in E1u:
            n2+=G1[edge[0]][edge[1]][attr]
        for edge in E2u:
            n2+=G2[edge[0]][edge[1]][attr]
        if n2 != 0:
            return n1/n2
        else:
            return None
    else:
        return None


# ## 3.1.2 Graph Edit Distance

# In[8]:


def ged(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    sim = len(V1)+len(V2)+len(E1)+len(E2)-2*(len(V1&V2) + len(E1&E2))
    return sim


# In[9]:


def ged_norm(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    sim = 1 - (len(V1&V2) + len(E1&E2))/(len(V1|V2) + len(E1|E2))
    return sim


# ## 3.1.3 Vertex-Edge Overlap

# In[11]:


def vertex_edge_overlap(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    V_overlap, E_overlap = V1&V2, E1&E2
    sim = 2*(len(V_overlap) + len(E_overlap)) / (len(V1)+len(V2)+len(E1)+len(E2))
    return sim


# ## 3.1.4 k-hop Nodes Neighborhood

# In[13]:


def get_neigbors(G, v, k):
    neighbors = set()
    k_hop_n = []
    for i in range(k):
        if i==0: 
            k_hop_n.append(set(G.neighbors(v)))
            neighbors |= set(k_hop_n[i])
        else:
            nn = set()
            for node in k_hop_n[i-1]:
                nn |= set(G.neighbors(node))
            k_hop_n.append(set(nn - neighbors - set([v])))
            neighbors |= k_hop_n[i]
    return neighbors  
        

def nodes_neighborhood(G1,G2, k):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    V_overlap = V1&V2
    sim = 0
    for v in V_overlap:
        n1, n2 = get_neigbors(G1, v, k), get_neigbors(G2, v, k)
        if len(n1|n2)>0:
            sim+=len(n1&n2)/len(n1|n2)
    return sim/(len(V1|V2))


# ## 3.1.5 Maximum Common Subgraph Distance (MCS)

# In[15]:


def getMCS(g1, g2):
    matching_graph=nx.Graph()
    mcs = 0

    for n1,n2 in g2.edges():
        if g1.has_edge(n1, n2):
            matching_graph.add_edge(n1, n2)
    if not nx.is_empty(matching_graph):
        components = (nx.connected_components(matching_graph))
        largest_component = max(components, key=len)
        mcs = len(largest_component)
    return mcs


# ## 3.1.7 Vector Similarity Algorithm 

# In[18]:


from scipy.stats import rankdata

def construct_vsgraph(G):
    Gm = nx.DiGraph()
    q = nx.pagerank(G)
    label = 'weight' if nx.is_weighted(G) else None
    degree = {}
    if isinstance(G, nx.Graph):
        degree = G.degree(weight=label) 
    else: 
        degree = G.out_degree(weight=label)
    
    #reconstruct graph
    E = list(G.edges(data=True))
    for edge in E:
        w = 1 if label is None else edge[2][label]
        if isinstance(G, nx.Graph):
            Gm.add_edge(edge[0], edge[1], weight=w*q[edge[0]]/degree[edge[0]])
            Gm.add_edge(edge[1], edge[0], weight=w*q[edge[1]]/degree[edge[1]])
        else: 
            Gm.add_edge(edge[0], edge[1], weight=w*q[edge[0]]/degree[edge[0]])
    return Gm
    
        
def compare_graph_weghts(G1,G2, attr="weight"):   
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    E_union = E1|E2
    sim = 0    
    for edge in E_union:
        if G1.has_edge(*edge):
            if G2.has_edge(*edge):
                sim+=abs(G1[edge[0]][edge[1]][attr]-G2[edge[0]][edge[1]][attr])/max(G1[edge[0]][edge[1]][attr],G2[edge[0]][edge[1]][attr])
            else:
                sim+=1
        else:
            sim+=1   
    return sim/len(E_union)
    

def vector_similarity(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    V_overlap = V1&V2
    num = len(V1|V2)
    G1m, G2m = construct_vsgraph(G1), construct_vsgraph(G2)
    return 1-compare_graph_weghts(G1m, G2m)


# ## 3.1.12 Vertex Ranking

# In[25]:


from scipy.stats import rankdata

def vertex_ranking(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    V_overlap = V1&V2
    num = len(V1|V2)
    #find maximal denominator
    d = 0
    for i in range(num):
        d += (i-(num-i-1))**2
    
    #compute centralities
    pi1,pi2 = nx.pagerank(G1), nx.pagerank(G2)
    for v in (V1-V_overlap):
        pi2[v]=0
    for v in (V2-V_overlap):
        pi1[v]=0
    ranks1, ranks2 = rankdata(list(pi1.values()), method='min'),rankdata(list(pi2.values()), method='min')
    
    #compute similarity
    sim = 0
    for i in range(num):
        sim+=(ranks1[i]-ranks2[i])**2

    return 1-2*sim/d


# ## 3.1.13 Degree Jenson-Shannon divergence

# In[27]:


from collections import Counter

def degree_vector_histogram(graph):
        """Return the degrees in both formats.

        max_deg is the length of the histogram, to be padded with
        zeros.

        """
        vec = np.array(list(dict(graph.degree()).values()))
        if next(nx.selfloop_edges(graph), False):
            max_deg = len(graph)
        else:
            max_deg = len(graph) - 1
        counter = Counter(vec)
        hist = np.array([counter[v] for v in range(max_deg+1)])
        return vec, hist

    
def degreeJSD(G1,G2):
    deg1, hist1 = degree_vector_histogram(G1)
    deg2, hist2 = degree_vector_histogram(G2)
    max_len = max(len(hist1), len(hist2))
    p1 = np.pad(hist1, (0, max_len - len(hist1)), 'constant', constant_values=0)
    p2 = np.pad(hist2, (0, max_len - len(hist2)), 'constant', constant_values=0)
    if sum(hist1)>0:
        p1 = p1/sum(p1)
    if sum(hist2)>0:
        p2 = p2/sum(p2)
    return netrd.utilities.entropy.js_divergence(p1,p2)**(1/2)
    


# ## 3.1.15 Communicability Sequence Entropy

# In[29]:


def create_comm_matrix(C, dictV):
    N = len(dictV)
    Ca = np.zeros((N, N))
    for v in C:
        for v2 in C[v]:
            Ca[dictV[v]][dictV[v2]] = C[v][v2]
    return Ca
    

def CommunicabilityJSD(G1, G2):
    dist = 0
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    N1, N2 = len(V1), len(V2)
    V = list(V1|V2)
    N = len(V)
    dictV = dict(zip(V, list(range(N))))
    
    Ca1 = create_comm_matrix(nx.communicability_exp(G1), dictV)
    Ca2 = create_comm_matrix(nx.communicability_exp(G2), dictV)

    lil_sigma1 = np.triu(Ca1).flatten()
    lil_sigma2 = np.triu(Ca2).flatten()

    big_sigma1 = sum(lil_sigma1[np.nonzero(lil_sigma1)[0]])
    big_sigma2 = sum(lil_sigma2[np.nonzero(lil_sigma2)[0]])

    P1 = lil_sigma1 / big_sigma1
    P2 = lil_sigma2 / big_sigma2
    P1 = np.array(sorted(P1))
    P2 = np.array(sorted(P2))

    dist = netrd.utilities.entropy.js_divergence(P1, P2)
    return dist


# ## 3.1.19 λ-distances

# In[34]:


def lambda_distances(G1, G2):
        d=dict()
        labels = ["λ-d Adj.","λ-d Lap.","λ-d N.L."]
        # Get adjacency matrices
        try:
            A1, A2 = nx.to_numpy_array(G1), nx.to_numpy_array(G2)
            # List of matrix variations
            l1 = [A1, laplacian(A1), laplacian(A1, normed=True)]
            l2 = [A2, laplacian(A2), laplacian(A2, normed=True)]

            for l in range(len(l1)):
                ev1 = np.abs(eigvalsh(l1[l]))
                ev2 = np.abs(eigvalsh(l2[l]))
                d[labels[l]]= np.linalg.norm(ev1 - ev2)      
        except Exception:
            for l in range(len(labels)):
                d[labels[l]]= "None"
        return d   


# ## 3.1.26 Signature similarity (SS)

# In[44]:


import hashlib
from scipy.spatial.distance import hamming

def compute_features(G):
    features = []
    q = nx.pagerank(G)
    Gm = construct_vsgraph(G)
    
    for row in q:
        features.append([str(row), q[row]])
        #features[(str(row))]=q[row]
    
    E = set(Gm.edges())
    for edge in E:
        t = str(edge[0])+"_"+str(edge[1])
        w = Gm[edge[0]][edge[1]]['weight'] 
        #features[t]=w
        features.append([t, w])   
    return features

def encode_fingerprint(text):
    return bin(int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16))[2:]

def text2hash(h):
    for i in range(len(h)):
        h[i][0]=encode_fingerprint(h[i][0])  
    return h

def get_fingerprint(h):
    arr = np.zeros(128)
    for i in range(len(h)):
        for j in range(len(h[i][0])):
            w = h[i][1] if h[i][0][j]=='1' else -h[i][1]
            arr[j] += w
    for j in range(len(arr)):
        if arr[j]>0:
            arr[j]=1
        else:
            arr[j]=0
    return arr
    
    

def signature_similarity(G1, G2):
    h1 = get_fingerprint(text2hash(compute_features(G1)))
    h2 = get_fingerprint(text2hash(compute_features(G2)))
    dist = 1 - hamming(h1,h2)/len(h1)
    
    return dist
    


# ## 3.1.28 LD-measure

# In[47]:


def get_transition_distr(G, node, v_dict):
    arr = np.zeros(len(v_dict))
    neighbors = list(G.neighbors(node))
    if len(neighbors)>0:
        for el in G.neighbors(node):
            arr[v_dict[el]] = 1
        arr =  len(neighbors)
    return arr


def node_distance(G):
    """
    Return an NxN matrix that consists of histograms of shortest path
    lengths between nodes i and j. This is useful for eventually taking
    information theoretic distances between the nodes.

    Parameters
    ----------
    G (nx.Graph): the graph in question.

    Returns
    -------
    out (np.ndarray): a matrix of binned node distance values.

    """

    N = G.number_of_nodes()
    a = np.zeros((N, N))

    dists = nx.shortest_path_length(G)
    for idx, row in enumerate(dists):
        counts = Counter(row[1].values())
        a[idx] = [counts[l] for l in range(1, N + 1)]

    return a / (N - 1)
    

def ld_measure(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    V_overlap = V1&V2
    V = list(V1|V2)
    N = len(V)
    d = N-len(V_overlap)
    v_dict = dict()
    for i in range(len(V)):
        v_dict[V[i]] = i
    
    #Compute Node Distance Distribution
    nd1 = dict(zip(G1.nodes(), node_distance(G1)))
    nd2 = dict(zip(G2.nodes(), node_distance(G2)))
    for v in V_overlap:
        #Compute transition matrix
        d1 = get_transition_distr(G1, v, v_dict)
        d2 = get_transition_distr(G1, v, v_dict)
        d += (netrd.utilities.entropy.js_divergence(nd1[v],nd2[v])**(1/2)+(netrd.utilities.entropy.js_divergence(d1,d2)**(1/2)))/2
    return d/N


# ## 3.1.29 SLRIC

# In[49]:


import sys 
import slric


# # 4. Validation

# In[52]:


def to_undirected(G):
    G0=nx.Graph()
    label = 'weight' if nx.is_weighted(G) else None
    E = list(G.edges(data=True))
    for edge in E:
        w = 1 if label is None else edge[2][label]
        if G0.has_edge(*edge[:2]):
            G0[edge[0]][edge[1]]['weight'] += w
        else:
            G0.add_edge(edge[0], edge[1], weight = w)
    return G0


# In[53]:


def graph_sim(G1,G2, name):
    sim_dict = dict()
    try:
        if name=="Jaccard Index":
            sim_dict[name] = jaccard(G1,G2)
        elif name=="Weighted Jaccard Index":
            sim_dict[name] = jaccard_weighted(G1,G2)
        elif name=="Graph Edit Distance":
            sim_dict[name] = ged(G1,G2)
        elif name=="Graph Edit Distance (norm)":
            sim_dict[name] = ged_norm(G1,G2)
        elif name=="veo":
            sim_dict[name] = vertex_edge_overlap(G1,G2)
        elif name=="vertex_ranking":
            sim_dict[name] = vertex_ranking(G1,G2)
        elif name=="Vector Similarity Algorithm":
            sim_dict[name] = vector_similarity(G1,G2)
        elif name=="Signature similarity":
            sim_dict[name] = signature_similarity(G1,G2)
        elif name=="Frobenius":
            dist_obj = netrd.distance.Frobenius()
            sim_dict[name] = dist_obj.dist(G1, G2)
            sim_dict["Frobenius_norm"]=sim_dict[name]/len(set(G1.nodes())|set(G2.nodes()))
        elif name=="Frobenius (Weighted)":
            sim_dict[name]= scipy.sparse.linalg.norm(nx.to_scipy_sparse_array(G1, weight='weight')-nx.to_scipy_sparse_array(G2, weight='weight'))
        elif name=="IpsenMikhailov":
            dist_obj = netrd.distance.IpsenMikhailov()
            sim_dict[name]=dist_obj.dist(G1, G2)
        elif name=="HammingIpsenMikhailov":
            dist_obj = netrd.distance.HammingIpsenMikhailov()
            sim_dict[name] = dist_obj.dist(G1, G2)
        elif name=="PolynomialDissimilarity":
            dist_obj = netrd.distance.PolynomialDissimilarity()
            sim_dict[name] = dist_obj.dist(G1, G2)
        elif name=="degreeJSD":
            sim_dict[name] = degreeJSD(G1, G2)**(1/2)
        elif name=="PortraitDivergence":
            dist_obj = netrd.distance.PortraitDivergence()
            sim_dict[name] = dist_obj.dist(G1, G2)
        elif name=="CommunicabilityJSD":
            sim_dict[name] = CommunicabilityJSD(to_undirected(G1), to_undirected(G2))
        elif name=="GraphDiffusion":
            dist_obj = netrd.distance.GraphDiffusion()
            sim_dict[name] = dist_obj.dist(G1, G2)
        elif name=="ResistancePerturbation":
            dist_obj = netrd.distance.ResistancePerturbation()
            sim_dict[name] = dist_obj.dist(G1.subgraph(max(nx.connected_components(G1), key=len)), G2.subgraph(max(nx.connected_components(G2), key=len)))
        elif name=="NetLSD":
            dist_obj = netrd.distance.NetLSD()
            sim_dict[name] = dist_obj.dist(G1, G2)
        elif name=="NetSimile":
            dist_obj = netrd.distance.NetSimile()
            sim_dict[name] = dist_obj.dist(G1, G2)
        elif name=="lambda":
            sim_dict = lambda_distances(G1, G2)
        elif name=="Lap.JS":
            dist_obj = netrd.distance.LaplacianSpectral()
            sim_dict[name] = dist_obj.dist(G1, G2)
        elif name=="NBD":
            dist_obj = netrd.distance.NonBacktrackingSpectral()
            sim_dict[name] = dist_obj.dist(G1, G2)  
        elif name=="d-NBD":
            dist_obj = netrd.distance.DistributionalNBD()
            sim_dict[name] = dist_obj.dist(G1, G2)  
        elif name=="Onion Spectrum (Extended)":
            dist_obj = netrd.distance.OnionDivergence()
            sim_dict[name] = dist_obj.dist(G1, G2)  
        elif name=="dk-series":
            dist_obj = netrd.distance.dkSeries()
            sim_dict[name] = dist_obj.dist(G1, G2)  
        elif name=="nodes_neighborhood":
            sim_dict[name+" 1"] = nodes_neighborhood(G1,G2, 1)
            sim_dict[name+" 2"] = nodes_neighborhood(G1,G2, 2)
            sim_dict[name+" 3"] = nodes_neighborhood(G1,G2, 3)
        elif name=="MCS":
            sim_dict[name] = 1-getMCS(G1,G2)/max(G1.number_of_nodes(),G2.number_of_nodes())
        elif name=="deltacon":
            dist_obj = netrd.distance.DeltaCon()
            sim_dict[name] = 1/(1+dist_obj.dist(G1, G2))  
        elif name=="d-measure":
            dist_obj = netrd.distance.DMeasure()
            sim_dict[name] = dist_obj.dist(G1.subgraph(max(nx.connected_components(G1), key=len)), G2.subgraph(max(nx.connected_components(G2), key=len)))
        elif name=="LD-measure":
            sim_dict[name] = ld_measure(G1.subgraph(max(nx.connected_components(G1), key=len)), G2.subgraph(max(nx.connected_components(G2), key=len)))
        elif name=="lric_sim":
            sim_dict[name] = np.linalg.norm(slric.graphsim(G1,G2))/(2)**(1/2)
        elif name=="QuantumJSD":
            dist_obj = netrd.distance.QuantumJSD()
            sim_dict[name] = dist_obj.dist(G1, G2)
    except Exception as e: 
        #print(e)
        sim_dict[name] = None   
    return sim_dict


# In[54]:


def graph_similarities(G1,G2):
    distance_list = ["Jaccard Index", "Weighted Jaccard Index", "Graph Edit Distance", "Graph Edit Distance (norm)",
                    "veo", "vertex_ranking", "Vector Similarity Algorithm", "Signature similarity", "Frobenius", 
                     "Frobenius (Weighted)", "IpsenMikhailov", "HammingIpsenMikhailov", "PolynomialDissimilarity",
                    "degreeJSD", "PortraitDivergence", "CommunicabilityJSD", "GraphDiffusion", "ResistancePerturbation",
                    "NetLSD", "NetSimile", "lambda", "Lap.JS", "NBD", "d-NBD", "Onion Spectrum (Extended)", "dk-series",
                    "nodes_neighborhood", "MCS", "deltacon", "d-measure", "LD-measure", "lric_sim", "QuantumJSD"]
    time_stamps = dict()
    sim_dict = dict()
    for dist_name in distance_list:
        sim_dict.update(graph_sim(G1,G2, dist_name))  
    return sim_dict


# In[55]:


distance_list = ["Jaccard Index", "Weighted Jaccard Index", "Graph Edit Distance", "Graph Edit Distance (norm)",
                    "veo", "vertex_ranking", "Vector Similarity Algorithm", "Signature similarity", "Frobenius", 
                     "Frobenius (Weighted)", "IpsenMikhailov", "HammingIpsenMikhailov", "PolynomialDissimilarity",
                    "degreeJSD", "PortraitDivergence", "CommunicabilityJSD", "GraphDiffusion", "ResistancePerturbation",
                    "NetLSD", "NetSimile", "lambda", "Lap.JS", "NBD", "d-NBD", "Onion Spectrum (Extended)", "dk-series",
                    "nodes_neighborhood", "MCS", "deltacon", "d-measure", "LD-measure", "lric_sim", "QuantumJSD"]


# # Simple Graphs

# In[59]:


ExamplesList=[]

#Example 1
G1_1=nx.Graph([(1, 2), (1, 3), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)])
G1_2=nx.Graph([(1, 2), (1, 3), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5)])
G1_3=nx.Graph([(1, 2), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5)])
ExamplesList.append([G1_1,G1_2,G1_3])

#Example 2
G2_1=nx.Graph([(1, 2), (2, 3), (3, 4), (4,5), (5,1)])
G2_2=nx.Graph([(1, 2), (2, 3), (3, 4), (5,1)])
G2_3=nx.Graph([(1, 2), (3, 4), (5,1)])
ExamplesList.append([G2_1,G2_2,G2_3])

#Example 3
G3_1=nx.Graph([(1, 2), (1, 3), (2, 3), (4,1), (5,1), (6,2), (7,2), (8,3), (9,3)])
G3_2=nx.Graph([(5, 1), (1, 4), (4, 6), (6,2), (6,7), (7,2), (2,3), (7,8), (3,8)])
G3_2.add_node(9)
G3_3=nx.Graph([(1, 4), (4, 5), (5, 1), (2,6), (6,7), (7,2), (3,8), (8,9), (9,3)])
ExamplesList.append([G3_1,G3_2,G3_3])

#Example 4
G4_1=nx.Graph([(1, 2), (1, 3), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5),
              (6, 7), (6, 8), (6, 9), (6,10), (7,8), (7,9), (7,10), (8,9), (8,10), (9,10),
              (5,6)])
G4_2=nx.Graph([(1, 2), (1, 3), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), 
              (6, 7), (6, 8), (6, 9), (6,10), (7,8), (7,9), (7,10), (8,9), (8,10), (9,10),
              (5,6)])
G4_3=nx.Graph([(1, 2), (1, 3), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5),
              (6, 7), (6, 8), (6, 9), (6,10), (7,8), (7,9), (7,10), (8,9), (8,10), (9,10)])
ExamplesList.append([G4_1,G4_2,G4_3])

#Example 5
G5_1=nx.Graph([(1, 2), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5),
              (6, 7), (6, 8), (6, 9), (6,10), (7,8), (7,10), (8,9), (8,10), (9,10),
              (1,10), (4,7)])
G5_2=nx.Graph([(1, 2), (1,3), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), 
              (6, 7), (6, 8), (6, 9), (6,10), (7,8), (7,9), (7,10), (8,9), (8,10), (9,10),
              (4,7)])
G5_3=nx.Graph([(1, 2), (1,3), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5),
              (6, 7), (6, 8), (6, 9), (6,10), (7,8), (7,9), (7,10), (8,9), (8,10), (9,10)])
ExamplesList.append([G5_1,G5_2,G5_3])

#Example 6
G6_1=nx.Graph([(1, 2), (1, 3), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5),
              (5,6),(6,7),(7,8),(8,9)])
G6_2=nx.Graph([(1, 2), (1, 3), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), 
              (5,6),(6,7),(7,8),(8,9)])
G6_3=nx.Graph([(1, 2), (1, 3), (1, 4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5),
              (6,7),(7,8),(8,9)])
ExamplesList.append([G6_1,G6_2,G6_3])


# In[67]:


dict(zip([1,2,3], ["a"]*3))


# In[76]:


res = []
for i in range(len(ExamplesList)):
    array = ["Example"+str(i+1),"G1","G2"]
    sim_dict = graph_similarities(ExamplesList[i][0],ExamplesList[i][1]) #1 vs.2
    for name in list(dict_labels):
        array.append(sim_dict[name])
    res.append(array)
    
    array = ["Example"+str(i+1),"G2","G3"]
    sim_dict = graph_similarities(ExamplesList[i][1],ExamplesList[i][2]) #2 vs.3
    for name in list(dict_labels):
        array.append(sim_dict[name])
    res.append(array)
    
    array = ["Example"+str(i+1),"G1","G3"]
    sim_dict = graph_similarities(ExamplesList[i][0],ExamplesList[i][2]) #1 vs.3
    for name in list(dict_labels):
        array.append(sim_dict[name])
    res.append(array)

res = pd.DataFrame(res, columns = ["Example", "Graph X", "Graph Y"] + list(dict_labels))
res.rename(dict_labels, axis=1, inplace = True)
res.to_excel('small_examples.xlsx')   


# ## Artificial Data

# In[55]:


def generate_new_graph(graph, ratio_addition, ratio_removal):
    G=graph.copy()
    edges = list(graph.edges)
    nonedges = list(nx.non_edges(graph))

    # random edge choice
    chosen_edge = random.sample(edges, min(len(edges), round(len(edges)*random.randint(0, ratio_removal)/100)))
    chosen_nonedge = random.sample(nonedges, min(len(nonedges), round(len(edges)*random.randint(0, ratio_addition)/100)))

    #remove edges
    G.remove_edges_from(chosen_edge)
    # add new edge
    G.add_edges_from(chosen_nonedge)

    return G


# In[ ]:


iterations = 1000
timestamps = 80
size_graph = 100
ratio_removal = 20 #percentages
ratio_addition = 25 #percentages
p = 2/(size_graph-1)

avg_corrK = None
avg_corrP = None
avg_corrS = None

for l in tqdm(range(iterations)):
    G1 = nx.fast_gnp_random_graph(size_graph, p)
    G2 = generate_new_graph(G1, ratio_addition, ratio_removal)
    res = []
    for time in range(timestamps): 
        if time != 0:
            G1=G2.copy()
            G2=generate_new_graph(G1, ratio_addition, ratio_removal)
        sim_dict = graph_similarities(G1,G2)
        array = []# [time,time+1]
        for name in list(sim_dict):
            v = sim_dict[name]
            array.append(v if v is None else v+0.00000000000001*(timestamps==time+1))
        res.append(array)
    sim_dataframe = pd.DataFrame(res, columns =  list(sim_dict))
    corrK = sim_dataframe.corr(method='kendall')
    corrP = sim_dataframe.corr(method='pearson')
    corrS = sim_dataframe.corr(method='spearman')
    avg_corrK = corrK if avg_corrK is None else avg_corrK+corrK
    avg_corrP = corrP if avg_corrP is None else avg_corrP+corrP
    avg_corrS = corrS if avg_corrS is None else avg_corrS+corrS
    if l%10==0:     
        avg_corrK.to_excel('avg_corrK'+str(l)+'.xlsx')
        avg_corrP.to_excel('avg_corrP'+str(l)+'.xlsx')
        avg_corrS.to_excel('avg_corrS'+str(l)+'.xlsx')
        

avg_corrK=avg_corrK/iterations
avg_corrP=avg_corrP/iterations
avg_corrS=avg_corrS/iterations
       
    


# In[56]:


dict_labels = {"Jaccard Index":"JI", "Weighted Jaccard Index": "wJI", "Graph Edit Distance": "GED", "Graph Edit Distance (norm)": "GED norm",
                    "veo":"VEO", "vertex_ranking": "VR", "Vector Similarity Algorithm": "VS", "Signature similarity": "SS", "Frobenius":"FRO", "Frobenius_norm":"FRO (norm)",
                     "Frobenius (Weighted)": "FRO weighted", "IpsenMikhailov": "IM", "HammingIpsenMikhailov": "HIM", "PolynomialDissimilarity": "POL",
                    "degreeJSD": "degreeJSD", "PortraitDivergence":"POR", "CommunicabilityJSD": "CSE", "GraphDiffusion": "GDD", "ResistancePerturbation": "RP",
                    "NetLSD": "NetLSD", "NetSimile": "NetSimile", "λ-d Adj.": "λ-d Adj.","λ-d Lap.":"λ-d Lap.",
               "λ-d N.L.":"λ-d N.L.", "Lap.JS":"Lap.JS", "NBD": "NBD", "d-NBD":"d-NBD", "Onion Spectrum (Extended)": "OnionS",
               "dk-series":"dk-series", "nodes_neighborhood 1": "1-hop NN", "nodes_neighborhood 2": "2-hop NN", 
               "nodes_neighborhood 3": "3-hop NN", "MCS": "MCS", "deltacon": "deltacon", "d-measure":"d-measure", 
               "LD-measure":"LD-measure", "lric_sim": "SLRIC-sim", "QuantumJSD":"QJSD"}
print(dict_labels)


# In[197]:


avg_corrK.rename(dict_labels, axis=0, inplace = True)
avg_corrK.rename(dict_labels, axis=1, inplace = True)
avg_corrK


# In[185]:


fig, ax = plt.subplots(figsize=(15,10))  # Sample figsize in inches

#avg_corrK.to_excel('avg_corrK.xlsx')

sb.heatmap(avg_corrK, 
            xticklabels=avg_corrK.columns,
            yticklabels=avg_corrK.columns,
            cmap='RdBu_r',
            annot=False,
            linewidth=0., ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches

#avg_corrP.to_excel('avg_corrP.xlsx')

sb.heatmap(avg_corrP, 
            xticklabels=avg_corrP.columns,
            yticklabels=avg_corrP.columns,
            cmap='RdBu_r',
            annot=False,
            linewidth=0., ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches

#avg_corrS.to_excel('avg_corrS.xlsx')

sb.heatmap(avg_corrS, 
            xticklabels=avg_corrS.columns,
            yticklabels=avg_corrS.columns,
            cmap='RdBu_r',
            annot=False,
            linewidth=0., ax=ax)


# In[156]:


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=15):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


# In[157]:


print("Top Absolute Correlations")
print(list(get_top_abs_correlations(avg_corrK, 700))[:144])


# In[295]:


data = list(get_top_abs_correlations(avg_corrK, 700))
# fixed bin size
bins = np.arange(0, 1.1, 0.025) # fixed bin size

plt.xlim([min(data), max(data)])

plt.hist(data, bins=bins, alpha=0.7, color = "black")
plt.title('Pairwise Correlation Between Graph Distance Measures')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.savefig('correlation_histogram.png', dpi=600)
plt.show()


# In[159]:


matrix = avg_corrK.dropna(how='all').dropna(axis=1,how='all').to_numpy()

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if not np.isclose(a[i,j], a[j,i]):
                print(i, j, a[i,j], a[j,i])

check_symmetric(matrix)


# In[289]:


import scipy.cluster.hierarchy as h
import scipy.spatial.distance as ssd
distance_matrix = 1 - np.abs(avg_corrK.dropna(how='all').dropna(axis=1,how='all'))
labels_dend = distance_matrix.columns
Z = h.linkage(h.distance.squareform(distance_matrix), 'average')
#distArray = ssd.squareform(distance_matrix) 
#corr_condensed = h.distance.squareform(distance_matrix)
#Z = h.linkage(corr_condensed, method='average', metric='euclidean')
labels_clusters = h.fcluster(Z, 0.47, criterion='distance')
print(labels_clusters)


# In[290]:


fig = plt.figure(figsize=(22, 8), dpi=1400)
dn = h.dendrogram(Z, labels = labels_dend, orientation='top', 
           leaf_rotation=90)
#plt.savefig('air_dendogram.png', dpi=600, bbox_inches='tight')
plt.show()


# In[291]:


#from sklearn.cluster import AgglomerativeClustering
#from sklearn.metrics import silhouette_score
#
#
#silhouette_coefficients = []
#max_clust = 20
#
#for k in range(2, max_clust):
#    cluster = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
#    cluster.fit_predict(distance_matrix)
#    score = silhouette_score(distance_matrix, cluster.labels_, metric = 'precomputed')
#    silhouette_coefficients.append(score)
#
#num_clusters = silhouette_coefficients.index(max(silhouette_coefficients))+2
#print("Number of clusters:", num_clusters)
#cluster = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
#cluster.fit_predict(distance_matrix)
#     
#    
##figure = plt.figure(figsize=(4, 2), dpi=200)
#plt.plot(range(2, max_clust), silhouette_coefficients)
#plt.xticks(range(2, max_clust))
#plt.xlabel("Number of Clusters")
#plt.ylabel("Silhouette Coefficient")
#plt.show()


# In[292]:


ordered_labels = []
clusters = labels_clusters #cluster.labels_[k]
max_clusters = max(clusters)
for i in range(max_clusters+1):
    if i in clusters:
        print("Cluster "+ str(i)+":")
    for k in range(len(avg_corrK.columns)):
        if clusters[k]==i:
            ordered_labels.append(avg_corrK.columns[k])
            print(avg_corrK.columns[k])


# In[293]:


avg_corrK=avg_corrK.reindex(ordered_labels, axis=0).reindex(ordered_labels, axis=1)


# In[294]:


fig, ax = plt.subplots(figsize=(15,10))  # Sample figsize in inches

#avg_corrK.to_excel('avg_corrK.xlsx')

sb.heatmap(abs(avg_corrK), 
            xticklabels=avg_corrK.columns,
            yticklabels=avg_corrK.columns,
            cmap='Greys',
            annot=False,
            linewidth=0., ax=ax)

plt.savefig('results.png', dpi=600)
plt.show()


# # Applications

# ## Air Transportation Data

# In[296]:


#open data on flights (monthly statistics)
df_flights = pd.read_csv(r"dataset.csv") 
#df_flights = pd.read_csv(r"C:\Users\shvydun\Downloads\Airports\dataset.csv") 
df_edges = df_flights.groupby(['Time','iso3_origin','iso3_destination'])['weight'].sum().reset_index(name='weight')
nodes = set(df_edges['iso3_origin']) | set(df_edges['iso3_destination'])
periods = sorted(set(df_edges['Time']))


# In[310]:


#create graphs
listG = []

res = []
for y in periods:
    edges = df_edges[df_edges['Time']==y]
    g = nx.from_pandas_edgelist(edges, source='iso3_origin', target='iso3_destination', edge_attr='weight', create_using=nx.DiGraph())
    g.add_nodes_from(nodes)
    listG.append(g)


# In[311]:


# get feature names
distance_labels = list(graph_similarities(listG[0],listG[0]))
print(distance_labels)


# In[312]:


#compute graph similarities
i, j = 0, 1
res = []
max_it = round((len(periods)*(len(periods)-1))/2) #len(periods)-1 #
for l in tqdm(range(max_it)):
    # do something
    sim_dict = graph_similarities(listG[i],listG[j])
    array = [periods[i],periods[j]]
    for name in distance_labels:
        array.append(sim_dict[name])
    res.append(array)
    #i+=1
    j+=1
    #j+=1
    if j==len(periods):
        i+=1
        j=i+1
    l+=1


# In[313]:


res = pd.DataFrame(res, columns = ["date 1", "date 2"] + distance_labels)
res


# In[314]:


kendcorr = res.corr(method='kendall')
kendcorr


# In[315]:


res.to_excel('airport_dist'+str(630)+'.xlsx')


# In[316]:


fig, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches

sb.heatmap(kendcorr, 
            xticklabels=kendcorr.columns,
            yticklabels=kendcorr.columns,
            cmap='RdBu_r',
            annot=False,
            linewidth=0., ax=ax)


# In[319]:


import scipy.cluster.hierarchy as h
import scipy.spatial.distance as ssd
distance_matrix = 1 - np.abs(kendcorr.dropna(how='all').dropna(axis=1,how='all'))
labels_dend = distance_matrix.columns
Z = h.linkage(h.distance.squareform(distance_matrix), 'average')
#distArray = ssd.squareform(distance_matrix) 
#corr_condensed = h.distance.squareform(distance_matrix)
#Z = h.linkage(corr_condensed, method='average', metric='euclidean')
labels_clusters = h.fcluster(Z, 0.65, criterion='distance')
print(labels_clusters)


# In[318]:


fig = plt.figure(figsize=(22, 8), dpi=1400)
dn = h.dendrogram(Z, labels = labels_dend, orientation='top', 
           leaf_rotation=90)
#plt.savefig('air_dendogram.png', dpi=600, bbox_inches='tight')
plt.show()


# In[320]:


ordered_labels = []
clusters = labels_clusters #cluster.labels_[k]
max_clusters = max(clusters)
for i in range(max_clusters+1):
    if i in clusters:
        print("Cluster "+ str(i)+":")
    for k in range(len(kendcorr.columns)):
        if clusters[k]==i:
            ordered_labels.append(kendcorr.columns[k])
            print(kendcorr.columns[k])


# In[323]:


kendcorr=kendcorr.reindex(ordered_labels, axis=0).reindex(ordered_labels, axis=1)
kendcorr.rename(dict_labels, axis=0, inplace = True)
kendcorr.rename(dict_labels, axis=1, inplace = True)


# In[324]:


fig, ax = plt.subplots(figsize=(15,10))  # Sample figsize in inches

#avg_corrK.to_excel('avg_corrK.xlsx')

sb.heatmap(abs(kendcorr), 
            xticklabels=kendcorr.columns,
            yticklabels=kendcorr.columns,
            cmap='Greys',
            annot=False,
            linewidth=0., ax=ax)

plt.savefig('results_air.png', dpi=600)
plt.show()


# In[ ]:





# # Enron DataSet

# In[53]:


df_enron = pd.read_csv(r"C:\Users\shvydun\Downloads\enron_mail\out.txt", sep=' ') 
df_enron['time'] = pd.to_datetime(df_enron['time'],unit='s').dt.date


# In[54]:


df_enron


# In[55]:


df_edges = df_enron.groupby(['time','from','to'])['asym'].sum().reset_index(name='weight')
nodes = set(df_edges['from']) | set(df_edges['to'])
periods = sorted(set(df_edges['time']))


# In[75]:


# get feature names
g0 = nx.from_pandas_edgelist(df_edges[df_edges['time']==periods[0]], source='from', target='to', edge_attr='weight', create_using=nx.DiGraph())
distance_labels = list(graph_similarities(g0,g0))
print(distance_labels)


# In[76]:


#compute graph similarities
i, j = 0, 1
res = []
max_it = len(periods)-1 #round((len(periods)*(len(periods)-1))/2)
for l in tqdm(range(max_it)):
    # do something
    g1 = nx.from_pandas_edgelist(df_edges[df_edges['time']==periods[i]], source='from', target='to', edge_attr='weight', create_using=nx.DiGraph())
    g2 = nx.from_pandas_edgelist(df_edges[df_edges['time']==periods[j]], source='from', target='to', edge_attr='weight', create_using=nx.DiGraph())
    g1.add_nodes_from(nodes)
    g2.add_nodes_from(nodes)
    
    sim_dict = graph_similarities(g1,g2)
    array = [periods[i],periods[j]]
    for name in distance_labels:
        array.append(sim_dict[name])
    res.append(array)
    i+=1
    j+=1
    #j+=1
    #if j==len(periods):
    #    i+=1
     #   j=i+1
    l+=1


# In[ ]:


#listG = []
#for y in periods:
#    edges = df_edges[df_edges['time']==y]
#    g = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr='weight', create_using=nx.DiGraph())
#    g.add_nodes_from(nodes)
#    listG.append(g)


# In[77]:


res = pd.DataFrame(res, columns = ["date 1", "date 2"] + distance_labels)
res


# In[78]:


pearsoncorr = res.corr(method='pearson')
pearsoncorr


# In[79]:


fig, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches

sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=False,
            linewidth=0., ax=ax)


# In[ ]:




