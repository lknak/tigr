import numpy as np
import torch
import torch.optim
import time
import rlkit.torch.pytorch_util as ptu


class PCGradOptimizer():

    def __init__(self, optimizers : [torch.optim], verbose=False):

        self.optimizers = optimizers if type(optimizers) is list else [optimizers]
        self.verbose = verbose

    def compute_gradients(self, losses, retain_graph):

        t0 = time.time()

        assert type(losses) is list
        num_tasks = len(losses)
        np.random.shuffle(losses)

        # Compute per-task gradients from losses
        t12 = time.time()
        task_gradients = []
        for loss_index, loss in enumerate(losses):
            # Reset all optimizers
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            loss.backward(retain_graph=((loss_index < len(losses)-1) or retain_graph))

            # Extract gradients for all optimizers, for all param groups, for all params
            optimizer_gradients = []
            for optimizer in self.optimizers:
                group_gradients = []
                for group in optimizer.param_groups:
                    param_gradients = []
                    for p in group['params']:
                        if p.grad is not None and p.requires_grad:
                            grad = p.grad
                            if grad.is_sparse:
                                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                            param_gradients.append(grad.data.clone().reshape(-1).to())
						
                        # Store zeros in case gradient is needed somewhere else (then we simply have a zero multiplication but full shape)
                        if p.grad is None and p.requires_grad:
                            param_gradients.append(torch.zeros_like(p.data).reshape(-1).to())
							
                    group_gradients.append(torch.cat(param_gradients))
                optimizer_gradients.append(torch.cat(group_gradients))
            task_gradients.append(torch.cat(optimizer_gradients))
        if self.verbose: print(f'\n\n----------\nBackprop: {time.time() - t12} s')

        # Stack task gradients
        task_gradients = torch.stack(task_gradients)

        # Compute per-task gradients.
        t1 = time.time()
        original_task_gradients = task_gradients.clone()
        zero_tensor = ptu.zeros(1)
        for k in range(num_tasks):
            # Parallel computation of projections using matrix multiplication
            inner_product = torch.matmul(task_gradients, original_task_gradients[k, :])
            proj_direction = torch.min(inner_product / (torch.matmul(original_task_gradients[k, :], original_task_gradients[k, :]) + 1e-12), zero_tensor)
            task_gradients -= (proj_direction * original_task_gradients[k, :][:, None]).t()
        if self.verbose: print(f'Computation projection: {time.time() - t1} s')

        # Assign projected gradients back to their params
        task_gradients = torch.sum(task_gradients, dim=0)

        t2 = time.time()
        # Reset optimizers so last gradients do not affect computation
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        start_idx = 0
        for optimizer in self.optimizers:
            for l, group in enumerate(optimizer.param_groups):
                for m, p in enumerate(group['params']):
                    if p.requires_grad:
                        grad_shape = p.shape
                        flatten_dim = np.prod(grad_shape)
                        proj_grad = task_gradients[start_idx:start_idx + flatten_dim]
                        # Direct assignment instead of assigning to object in case it is copied
                        if p.grad is None:
                            # In case the grad property is still None, we have to assign the zeros that we stored at the beginning
                            optimizer.param_groups[l]['params'][m].grad = proj_grad.reshape(grad_shape)
                        else:
                            optimizer.param_groups[l]['params'][m].grad += proj_grad.reshape(grad_shape)

                        start_idx += flatten_dim
        if self.verbose: print(f'Assignment: {time.time() - t2} s')

        if self.verbose: print(f'Total --> {time.time() - t0} s\n----------\n\n')

    def minimize(self, losses, retain_graph=False):
        self.compute_gradients(losses=losses, retain_graph=retain_graph)
        for optimizer in self.optimizers:
            optimizer.step()