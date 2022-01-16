import networkx as nx
import numpy as np
import random as r
import matplotlib.pyplot as plt
def try_basic_api():
    N = 10
    E = 20

    arr = np.random.randint(0,N,(E,2))
    # print(arr)

    G = nx.Graph()
    G.add_edges_from(arr)

    for e in G.edges:
        G.edges[e]['weight'] = r.randint(1,100)
        print(e)
        print(G.edges[e])
    #
    nx.draw(G,with_labels=True)
    plt.show()

    print(G.edges.data())

def draw_K_3_5():
    K_3_5 = nx.complete_bipartite_graph(3, 5)
    print(K_3_5.edges.data())
    pos = np.random.randint(0,100,(8,2))
    pos[0:3,1] = 5
    pos[3:,1] = 20
    print(pos)
    nx.draw(K_3_5,pos=pos,with_labels=True)
    plt.show()

def draw_barbell():
    # barbell图是指 两团网络，中间通过一条路径进行连接
    # 其中，第一个参数是团网络的节点个数，第二个参数是路径的大小
    barbell = nx.barbell_graph(10, 5)
    nx.draw(barbell)
    plt.show()

er = nx.erdos_renyi_graph(16, 0.2)
ws = nx.watts_strogatz_graph(30, 3, 0.1)
ba = nx.barabasi_albert_graph(100, 5)
red = nx.random_lobster(100, 0.9, 0.9)
g = er
print(er.number_of_edges())

nx.draw(g)
plt.show()