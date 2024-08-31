import torch
from experiment_utilities.memory import TreeMemory
from experiment_utilities.trees import tree_flatten, tree_fill, tree_map, child_is_leaf

example_tree = [
    {"foo": (torch.FloatTensor([3.5]),
             torch.LongTensor([2, 3])),
     "bar": [torch.BoolTensor([[True]]), {1: torch.FloatTensor([34.]),
                 's': torch.FloatTensor([6., 2.])}]
     },
    [torch.LongTensor([3]), torch.LongTensor([-3]), (torch.FloatTensor([3.2]), torch.FloatTensor([3.1]))]
]

def test_tree_memory():

    memory = TreeMemory(size=10)
    memory.store(example_tree)
    pass