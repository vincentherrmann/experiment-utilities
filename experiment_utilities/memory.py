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
    def __init__(self, size, shape_tree, type_tree=None, device=None):
        # artifact dict: dictionary of the artifacts to be stored in memory of the form {"name": shape, ...}
        # size the maximum size of the memory
        super().__init__()
        shapes, structure = tree_flatten(shape_tree, is_leaf=self.is_leaf)
        self.structure = structure
        self.shapes = shapes
        self.num_leaves = len(shapes)

        if type_tree is None:
            type_tree = tree_fill([torch.float] * self.num_leaves, structure=structure)

        self.memory_tree = tree_map(lambda shape, type: torch.zeros([size] + list(shape), dtype=type, device=device),
                                    tree=shape_tree, rest=[type_tree], is_leaf=self.is_leaf)

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, data_tree):
        tree_modify(self.store_leave, tree=self.memory_tree, rest=[data_tree])

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_multiple(self, data_tree):
        n = tree_reduce(lambda x, y: y.shape[0], tree=data_tree)
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

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
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