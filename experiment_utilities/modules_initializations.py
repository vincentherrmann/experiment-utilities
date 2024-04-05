import torch
from experiment_utilities.trees import tree_reduce


class OrderedSet:
    def __init__(self):
        self.items = []
        self.set = set()

    def add(self, item):
        if item not in self.set:
            self.items.append(item)
            self.set.add(item)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


def lsuv_init(model: torch.nn.Module,
              data: torch.Tensor,
              tolerance=1e-2,
              max_iter=10,
              recurrence_outputs='last',
              target_std=1.,
              decay=0.9):
    """
    Layer-sequential unit-variance initialization. Iteratively scales the weights of the model so that the output of
    each module has a standard deviation of target_std. The mean of the output is also set to 0.
    :param model: The torch module to initialize
    :param data: A representative batch of input data for the model
    :param tolerance: The tolerance for the standard deviation and mean of the output
    :param max_iter: The maximum number of iterations to run the initialization
    :param recurrence_outputs: If there are multiple outputs from a module due to recurrence,
    which one to use for the LSUV init. Options are 'all', 'last', 'first'
    :decay: The decay factor for the scaling of the weights
    """
    leaf_modules = [m for m in model.modules() if len(list(m.parameters(recurse=False))) > 0]
    ordered_modules = OrderedSet()

    def order_registration_hoook(module, args, output):
        ordered_modules.add(module)

    for module in leaf_modules:
        module.register_forward_hook(order_registration_hoook)

    # run the model on the data
    with torch.no_grad():
        model(data)

    for module in leaf_modules:
        module._forward_hooks.clear()

    for module in ordered_modules:
        print(f"Processing module {module}")

        for iteration in range(max_iter):
            parameters = module.named_parameters(recurse=False)
            outputs = []
            def get_output_hook(module, input, output):
                outputs.append(output)
            handle = module.register_forward_hook(get_output_hook)
            with torch.no_grad():
                model(data)
            handle.remove()

            if recurrence_outputs == 'all':
                output = outputs
            elif recurrence_outputs == 'last':
                output = [outputs[-1]]
            elif recurrence_outputs == 'first':
                output = [outputs[0]]

            flattened_output = tree_reduce(lambda x, y: torch.cat([x.flatten(), y.flatten()], dim=0), output)
            outputs_mean = flattened_output.mean()
            outputs_std = flattened_output.std()
            if iteration == 0:
                print(f"Before---outputs mean: {outputs_mean}, std: {outputs_std}")

            if abs(outputs_std - target_std) < tolerance and abs(outputs_mean) < tolerance:
                break
            current_decay = decay ** iteration
            for key, parameter in parameters:
                if "bias" in key:
                    parameter.data -= (outputs_mean * current_decay)
                else:
                    parameter.data *= 1 + ((target_std / outputs_std) - 1) * current_decay
        print(f"After---outputs mean: {outputs_mean}, std: {outputs_std}")