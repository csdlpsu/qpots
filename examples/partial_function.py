import torch

def multitask_multiobjective(X, task_id, n_tasks=3):
    """
    Multi-task multi-objective synthetic function.
    Here, each task_id corresponds to ONE objective.

    Args:
        X (torch.Tensor): shape (n, d), inputs in [0,1]^d
        task_id (int): which task/objective to evaluate (0 ... n_tasks-1)
        n_tasks (int): total number of tasks/objectives

    Returns:
        torch.Tensor: shape (n, 1), function values for that task/objective
    """
    d = X.shape[-1]
    torch.manual_seed(task_id)  # reproducible params per task

    # Task-dependent shifts and frequencies
    a = torch.linspace(0.2, 0.8, d) + 0.1 * task_id
    b = 2.0 + task_id + torch.arange(1, d+1) * 0.5
    c = 1.0 + 0.3 * task_id + torch.arange(1, d+1) * 0.2
    a, b, c = a.to(X), b.to(X), c.to(X)

    # Define one objective per task_id
    if task_id % 3 == 0:
        y = torch.sum((X - a) ** 2, dim=-1)             # quadratic
    elif task_id % 3 == 1:
        y = torch.sum(torch.sin(b * X), dim=-1)         # sine
    else:
        y = torch.sum(torch.cos(c * X), dim=-1)         # cosine

    return y.unsqueeze(-1)
