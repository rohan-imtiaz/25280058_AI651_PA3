import torch
import torch.nn as nn

from common import graph_utils


class TemporalEncoder(nn.Module):
    def __init__(self, input_length, hidden_dimension):
        super().__init__()
        self.convolution_one = nn.Conv1d(1, hidden_dimension, kernel_size = 3)
        self.convolution_two = nn.Conv1d(hidden_dimension, hidden_dimension, kernel_size = 3)
        remaining_length = input_length - 4 # Two kernels of size 3 remove 2 steps each
        self.projection = nn.Linear(hidden_dimension * remaining_length, hidden_dimension)

    def forward(self, x):
        x = torch.relu(self.convolution_one(x))
        x = torch.relu(self.convolution_two(x))
        x = x.flatten(start_dim = 1)
        return torch.relu(self.projection(x))


class SpatioTemporalModel(nn.Module):
    def __init__(self, adjacency_mode, fixed_adjacency, number_of_nodes, input_length, output_length,
                 hidden_dimension, embedding_dimension, top_k, learn_alpha, use_skip_connection):
        super().__init__()
        self.adjacency_mode = adjacency_mode # One of none, fixed, adaptive, hybrid
        self.top_k = top_k
        self.use_skip_connection = use_skip_connection
        self.encoder = TemporalEncoder(input_length, hidden_dimension)
        self.graph_weight = nn.Linear(hidden_dimension, hidden_dimension, bias = False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, output_length),
        )
        if adjacency_mode in ["fixed", "hybrid"]:
            symmetric_form, _ = graph_utils.normalise_fixed_adjacency(fixed_adjacency)
            self.register_buffer("fixed_normalised", torch.tensor(symmetric_form, dtype = torch.float32))
            # The hybrid graph uses the row form so it is consistent with the row softmax of the adaptive graph
            row_form = graph_utils.row_normalise(fixed_adjacency)
            self.register_buffer("fixed_row_normalised", torch.tensor(row_form, dtype = torch.float32))
        if adjacency_mode in ["adaptive", "hybrid"]:
            self.source_embedding = nn.Parameter(torch.randn(number_of_nodes, embedding_dimension))
            self.target_embedding = nn.Parameter(torch.randn(number_of_nodes, embedding_dimension))
        if adjacency_mode == "hybrid":
            # sigmoid(0) = 0.5 so alpha starts at 0.5, and it is frozen when learn_alpha is False
            self.alpha_logit = nn.Parameter(torch.zeros(1), requires_grad = learn_alpha)

    def get_adaptive_adjacency(self):
        scores = torch.relu(self.source_embedding @ self.target_embedding.T)
        if self.top_k is None:
            return torch.softmax(scores, dim = 1)
        # Top k is applied before the softmax: the kept scores go through the softmax and the rest get weight 0
        # Gradients reach the embeddings only through the kept scores, the choice of indices has no gradient
        top_values, top_indices = torch.topk(scores, self.top_k, dim = 1)
        masked = torch.full_like(scores, float("-inf"))
        masked = masked.scatter(1, top_indices, top_values)
        return torch.softmax(masked, dim = 1)

    def get_alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def get_adjacency(self):
        if self.adjacency_mode == "fixed":
            return self.fixed_normalised
        if self.adjacency_mode == "adaptive":
            return self.get_adaptive_adjacency()
        alpha = self.get_alpha()
        return alpha * self.fixed_row_normalised + (1 - alpha) * self.get_adaptive_adjacency()

    def forward(self, x):
        # x has shape batch, time, nodes
        batch_size, time_length, number_of_nodes = x.shape
        sequences = x.permute(0, 2, 1).reshape(batch_size * number_of_nodes, 1, time_length)
        z = self.encoder(sequences).view(batch_size, number_of_nodes, -1)
        if self.adjacency_mode == "none":
            h = z
        else:
            adjacency = self.get_adjacency()
            h = torch.relu(self.graph_weight(torch.matmul(adjacency, z)))
            if self.use_skip_connection:
                h = h + z # Keeps the own sensor signal that neighbour averaging washes out
        prediction = self.head(h)
        return prediction.permute(0, 2, 1)