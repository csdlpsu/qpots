import warnings
import os

warnings.filterwarnings('ignore')

from qpots.acquisition import Acquisition
from qpots.model_object import ModelObject
from qpots.function import Function

import torch
from botorch.utils.transforms import unnormalize

device = torch.device("cpu")
args = dict(
        {
            "ntrain": 20,
            "iters": 50,
            "reps": 20,
            "q": 1,
            "wd": ".",
            "ref_point": torch.tensor([-300, -18]),
            "dim": 2,
            "nobj": 2,
            "ncons": 0,
            "nystrom": 0,
            "nychoice": "pareto",
            "ngen": 10,
        }
    )

tf = Function('branincurrin', dim=args["dim"])
f = tf.evaluate
bounds = tf.get_bounds()

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(1023)

train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double)
train_y = f(unnormalize(train_x, bounds))

gps = ModelObject(train_x=train_x, train_y=train_y, bounds=bounds, nobj=args["nobj"], ncons=args["ncons"], device=device)
gps.fit_gp()

acq = Acquisition(tf, gps, cons=None, device=device, q=args["q"])
x, y, times, hv, pf = acq.tsemo(args["wd"], args["iters"], args["ref_point"], args["ntrain"], args["reps"])
print(f"Candidates: {x}, HVs: {hv}")