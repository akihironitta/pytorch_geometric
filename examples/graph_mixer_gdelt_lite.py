import torch

from torch_geometric.datasets import GDELTLite
from torch_geometric.loader import LinkNeighborLoader

dataset = GDELTLite(root="./")
data = dataset[0]

print(data.x)
assert data.x.size() == (8_831, 413), f"got x={data.x.size()}"
assert data.edge_index.size() == (2,
                                  1_912_909), f"got {data.edge_index.size()}"
assert data.time.size() == (1_912_909, ), f"got time={data.time}"
assert data.edge_attr.size() == (1_912_909,
                                 186), f"got edge_attr={data.edge_attr.size()}"

data.validate()
batch_size = 1
loader = LinkNeighborLoader(
    data=data,
    num_neighbors=[2, 2],
    batch_size=batch_size,
    edge_label_index=None,  # sample from all edges
    edge_label=data.edge_label,  # same length as `edge_label_index`
    edge_label_time=data.time,  # same length as `edge_label_index`
    temporal_strategy="last",
    time_attr="time",
    # directed=False,  # unsupported?
)

print("=============================================================")
data = next(iter(loader))
print(data)

print("input edge index", data.edge_index.size())  # []

# [2, num_sampled_edges]
print("output edge index", data.edge_label_index.size())
# [num_sampled_edges, 186]
print("output edge label", data.edge_label.size())
# [num_sampled_edges,]
print("edge timestamp", data.time.size())

num_sampled_edges = len(data.edge_index)
assert (num_sampled_edges >=
        batch_size), "sampled edges are (almost) always larger than batch_size"


def model(x, edge_index, time) -> torch.Tensor:
    # === graph mixer ===
    # 1. encode link
    # 1.1. temporal encoding
    # 1.2. 1-layer MLP-mixer
    # 2. encode node
    # 3. classify link
    print(
        f"... model(x={x.size()}, edge_index={edge_index.size()}, time={time.size()})"
    )
    return torch.zeros((num_sampled_edges, 186))


def loss_fn(edge_label_out, edge_label) -> torch.Tensor:
    print(
        f"... loss_fn(edge_label_out={edge_label_out.size()}, edge_label={edge_label.size()})"
    )
    return torch.zeros((1, ))


edge_label_out = model(data.x, data.edge_index, data.time)
loss = loss_fn(edge_label_out[:batch_size], data.edge_label[:batch_size])
