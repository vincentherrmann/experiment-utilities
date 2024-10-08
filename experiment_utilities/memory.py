import torch
import numpy as np

from experiment_utilities.trees import tree_flatten, tree_map, tree_fill, tree_modify, tree_reduce

class GeneralMemory(torch.nn.Module):
    """
    FIFO memory buffer for any number of tensors
    """

    def __init__(self, size, artifact_shapes, artifact_types=None, device=None):
        # artifact dict: dictionary of the artifacts to be stored in memory of the form {"name": shape, ...}
        # size the maximum size of the memory

        self.memory_dict = {}
        for key, shape in artifact_shapes.items():
            if artifact_types is None or not key in artifact_types.keys():
                data_type = torch.float32
            else:
                data_type = artifact_types[key]
            self.memory_dict[key] = torch.zeros([size] + list(shape), dtype=data_type, device=device)

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, data_dict):
        for key, data in data_dict.items():
            self.memory_dict[key][self.ptr] = data.detach()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        batch_dict = {}

        for key, buffer in self.memory_dict.items():
            batch_dict[key] = buffer[idxs]
        return batch_dict

    def get_sequence(self, idx, length=None):
        assert idx <= self.size

        mem_dict = {}
        if self.ptr < self.size:
            start = idx - length
            if start >= 0:
                for key, buffer in self.memory_dict.items():
                    mem_dict[key] = buffer[start:idx]
            else:
                for key, buffer in self.memory_dict.items():
                    mem_dict[key] = torch.cat([buffer[start:], buffer[:idx]], dim=0)
        else:
            start = 0 if length is None else max(0, idx - length)
            for key, buffer in self.memory_dict.items():
                mem_dict[key] = buffer[start:idx]

        return mem_dict

    def get_whole_memory(self):
        mem_dict = {}

        for key, buffer in self.memory_dict.items():
            mem_dict[key] = buffer[:self.size]
        return mem_dict


class TreeMemory(torch.nn.Module):
    def __init__(self, size, shape_tree=None, type_tree=None, device=None):
        # artifact dict: dictionary of the artifacts to be stored in memory of the form {"name": shape, ...}
        # size the maximum size of the memory
        super().__init__()

        if shape_tree is not None:
            shapes, structure = tree_flatten(shape_tree, is_leaf=self.is_leaf)
            self.structure = structure
            self.shapes = shapes
            self.num_leaves = len(shapes)

            if type_tree is None:
                type_tree = tree_fill([torch.float] * self.num_leaves, structure=structure)

            self.memory_tree = tree_map(lambda shape, type: torch.zeros([size] + list(shape), dtype=type, device=device),
                                        tree=shape_tree, rest=[type_tree], is_leaf=self.is_leaf)
        else:
            self.memory_tree = None
        self.device = device
        self.ptr, self.size, self.max_size = 0, 0, size

    def initialize_memory(self, example_tree):
        shape_tree = tree_map(lambda x: x.shape, tree=example_tree)
        shapes, structure = tree_flatten(shape_tree, is_leaf=self.is_leaf)
        self.structure = structure
        self.shapes = shapes
        self.num_leaves = len(shapes)

        type_tree = tree_map(lambda x: x.dtype, tree=example_tree)
        self.memory_tree = tree_map(lambda shape, type: torch.zeros([self.max_size] + list(shape),
                                                                    dtype=type, device=self.device),
                                    tree=shape_tree, rest=[type_tree], is_leaf=self.is_leaf)
        pass

    def store(self, data_tree):
        if self.memory_tree is None:
            self.initialize_memory(data_tree)

        tree_modify(self.store_leave, tree=self.memory_tree, rest=[data_tree])

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_multiple(self, data_tree):
        if self.memory_tree is None:
            self.initialize_memory(tree_map(lambda x: x[0], tree=data_tree))

        n = tree_reduce(lambda x, y: y.shape[0], tree=data_tree)
        if type(n) is not int:
            # if there is only one leaf, the reduce function returns the leaf itself
            n = n.shape[0]
        if self.ptr + n >= self.max_size:
            n1 = self.max_size - self.ptr
            tree_1 = tree_map(lambda x: x[:n1], tree=data_tree)
            tree_modify(self.store_multiple_leaves, tree=self.memory_tree, rest=[tree_1])
            self.ptr = 0
            tree_2 = tree_map(lambda x: x[n1:], tree=data_tree)
            tree_modify(self.store_multiple_leaves, tree=self.memory_tree, rest=[tree_2])
            self.ptr = n - n1
        else:
            tree_modify(self.store_multiple_leaves, tree=self.memory_tree, rest=[data_tree])
            self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def store_leave(self, memory, data):
        memory[self.ptr] = data.detach()

    def store_multiple_leaves(self, memory, data):
        n = data.shape[0]
        memory[self.ptr:self.ptr + n] = data.detach()

    def reset(self):
        tree_modify(lambda x: x.zero_(), tree=self.memory_tree)
        self.ptr, self.size = 0, 0

    def sample_batch(self, batch_size=32, idxs=None, without_replacement=False):
        if idxs is None:
            if without_replacement:
                idxs = np.random.choice(self.size, size=batch_size, replace=False)
            else:
                idxs = np.random.randint(0, self.size, size=batch_size)
        batch_tree = tree_map(lambda memory: memory[idxs], tree=self.memory_tree)
        return batch_tree

    def get_sequence(self, idx, length=None):
        assert idx <= self.size
        if length is not None:
            assert length <= self.size

        mem_dict = {}
        if self.ptr < self.size:
            if length is None:
                length = self.size
            start = idx - length
            if start >= 0:
                seq_tree = tree_map(lambda memory: memory[start:idx], tree=self.memory_tree)
            else:
                seq_tree = tree_map(lambda memory: torch.cat([memory[start:], memory[:idx]], dim=0),
                                    tree=self.memory_tree)
        else:
            start = 0 if length is None else max(0, idx - length)
            seq_tree = tree_map(lambda memory: memory[start:idx], tree=self.memory_tree)

        return seq_tree

    def get_whole_memory(self):
        return self.get_sequence(self.size)

    @staticmethod
    def is_leaf(x):
        return (type(x) is tuple or type(x) is list) and type(x[0]) is int


class WeightedTreeMemory(TreeMemory):
    def __init__(self, size, shape_tree=None, type_tree=None, device=None, compute_weights=None):
        super().__init__(size=size, shape_tree=shape_tree, type_tree=type_tree, device=device)
        if compute_weights is None:
            compute_weights = self.compute_exponential_recency_weights
        self.compute_weights = compute_weights
        self.weights = torch.zeros(size, device=device)

    def store(self, data_tree):
        super().store(data_tree)
        self.weights = self.compute_weights(self)

    def store_multiple(self, data_tree):
        super().store_multiple(data_tree)
        self.weights = self.compute_weights(self)

    def reset(self):
        super().reset()
        self.weights.zero_()

    def initialize_memory(self, example_tree):
        super().initialize_memory(example_tree)
        self.weights.zero_()

    def sample_batch(self, batch_size=32, idxs=None, without_replacement=False):
        if idxs is None:
            if without_replacement:
                effective_weights = weights.clone()
                idxs = []
                for _ in range(batch_size):
                    idx = torch.distributions.Categorical(probs=effective_weights).sample()
                    idxs.append(idx)
                    effective_weights[idx] = 0.
                    effective_weights = effective_weights / effective_weights.sum()
                idxs = torch.tensor(idxs, device=self.device)
            else:
                idxs = torch.distributions.Categorical(probs=self.weights).sample((batch_size,))
        batch_tree = tree_map(lambda memory: memory[idxs], tree=self.memory_tree)
        return batch_tree

    @staticmethod
    def compute_exponential_recency_weights(tree_memory, gamma=0.99):
        size = tree_memory.size
        # exponential decay with gamma
        time_from_now = torch.arange(size, dtype=torch.float, device=tree_memory.device).flip(0)
        weights = torch.pow(gamma, time_from_now.float())
        weights = weights / weights.sum()

        if tree_memory.ptr != tree_memory.size:
            weights = torch.cat([weights[-tree_memory.ptr:], weights[:-tree_memory.ptr]], dim=0)
        elif tree_memory.size != tree_memory.max_size:
            weights = torch.cat([weights, torch.zeros(tree_memory.max_size - weights.shape[0],
                                                      device=tree_memory.device)], dim=0)
        return weights


class TreeMemoryDataset(torch.utils.data.Dataset, TreeMemory):
    # this class is a TreeMemory wrapped as a dataset
    def __init__(self, size, shape_tree, type_tree=None, device=None):
        TreeMemory.__init__(self, size=size, shape_tree=shape_tree, type_tree=type_tree, device=device)
        torch.utils.data.Dataset.__init__(self)

    def __getitem__(self, idx):
        return self.sample_batch(idxs=idx)

    def __len__(self):
        return self.size