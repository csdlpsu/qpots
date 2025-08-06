import warnings
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

warnings.filterwarnings('ignore')

from qpots.acquisition import Acquisition
from qpots.model_object import ModelObject
from qpots.utils.utils import expected_hypervolume

import torch
from qpots.function import Function
from botorch.utils.transforms import unnormalize

device = torch.device("cpu")
#Added mt as multi-task to args, 0 is false 1 is true
args = dict(
    {
        "ntrain": 10000,
        "iters": 300,
        "reps": 20,
        "q": 1,
        "wd": "..",
        "ref_point": torch.tensor([-10., -10.]),
        "dim": 2,
        "nobj": 2,
        "ncons": 0,
        "nystrom": 0,
        "nychoice": "pareto",
        "ngen": 10,
        "mt": 0,
    }
)

# Set up problem
tf = Function('zdt3', dim=args["dim"], nobj=args["nobj"])
f = tf.evaluate
bounds = tf.get_bounds()
#print("Bounds:\n",bounds)

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(1023)

# set up the training points
train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double)
train_y = -1*f(unnormalize(train_x, bounds)).numpy()
train_x=train_x.numpy()
fig = plt.figure(figsize = (6, 6))

pf_indices = NonDominatedSorting().do(train_y, only_non_dominated_front=True)
pareto_front = train_y[pf_indices]

plt.scatter(train_y[:, 0], train_y[:, 1], color='blue', label='All Points MultiTask', alpha=0.1)
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label='Pareto Front MultiTask')

plt.xlabel("f1")
plt.ylabel("f2")

plt.grid(True)
plt.legend()
plt.show()