import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(f'DataSet: {dataset}:')
print('='*10)
print(f'Number of graphs : {len(dataset)}')
print(f'Number of features: {dataset.num_node_features}')
print(f'Number of classes : {dataset.num_classes}')

data = dataset[0]
print('='*50)

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of edge_features : {data.num_edge_features}')
print(f'Average node degrees : { (2* data.num_edges) / data.num_nodes:.2f}')
print(f'Number of training nodes : {data.train_mask.sum()}')
print(data.has_self_loops(),data.has_isolated_nodes(),data.is_undirected())

print(type(data.edge_index))


# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                       f'Training Accuracy: {accuracy["train"]*100:.2f}% \n'
                       f' Validation Accuracy: {accuracy["val"]*100:.2f}%'),
                       fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()

from torch_geometric.utils import to_networkx

G = to_networkx(data,to_undirected=True)
visualize(G,color=data.y)

from torch.nn import Linear
from torch_geometric.nn import GCNConv

input_dim = 34
hidden_dim = 4

num_layers = 3

import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN,self).__init__()
        torch.manual_seed(12345)
        self.classifier = Linear(2,dataset.num_classes)
        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))


        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, 2))

    def forward(self, x, edge_index):

        for i in range(num_layers):
            x = self.convs[i](x,edge_index)
            x = x.tanh()

        embeddings = x

        # Apply a final (linear) classifier.
        out = self.classifier(embeddings)

        return out, embeddings

import time

model = GCN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr= 0.01)

def train(data):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    predicted_classes = torch.argmax(out, axis=1)
    target_classes = data.y

    acc = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float())

    return  loss, h, {'val':acc,'train':0.0}


# optimizer.zero_grad()
# out, h = model(data.x, data.edge_index)
# loss = criterion(out[data.train_mask], data.y[data.train_mask])
# loss.backward()
# optimizer.step()


for epoh in range(200):
    los,h,acc = train(data)
    if (epoh % 10 ==0):
        visualize(h,color=data.y,epoch=epoh,loss=los,accuracy=acc)
        time.sleep(0.3)