"""Run coupled, constrained qPOTS on the WeldedBeam benchmark."""

import torch

from qpots import Function, QPOTSConfig, QPOTSRunner

problem = Function("weldedbeam", dim=4, nobj=2)
config = QPOTSConfig(
    n_initial=40,
    iterations=50,
    batch_size=2,
    n_constraints=4,
    generations=10,
    multitask=True,
    seed=1023,
)


def report(result):
    print(
        f"Iteration {result.iteration}: candidates={result.candidate_x}; "
        f"observed shape={tuple(result.observed_values.shape)}"
    )


result = QPOTSRunner(problem, config, callbacks=[report]).run()
torch.save(result.train_x.cpu(), "weldedbeam_train_x.pt")
torch.save(result.train_y.cpu(), "weldedbeam_train_y.pt")
