import torch
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem

#Multi-Fidelity Currin
class MultiFidelityCurrin(MultiObjectiveTestProblem):
    dim = 2
    continuous_inds = list(range(2))
    num_objectives = 2  # HF and LF
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _ref_point = [0.0, 0.0]  # Set something meaningful later!

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
            dtype: The dtype that is used for the bounds of the function.
        """
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    @staticmethod
    def _currin_hf(X: torch.Tensor) -> torch.Tensor:
        x0 = X[..., 0]
        x1 = X[..., 1]
        factor1 = 1 - torch.exp(-1 / (2 * x1))
        numer = 2300 * x0.pow(3) + 1900 * x0.pow(2) + 2092 * x0 + 60
        denom = 100 * x0.pow(3) + 500 * x0.pow(2) + 4 * x0 + 20
        return factor1 * numer / denom

    def _currin_lf(self, X: torch.Tensor) -> torch.Tensor:
        hf = self._currin_hf(X)
        return 0.5 * hf + (X[..., 0] - 0.5) * (X[..., 1] - 0.5)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        hf = self._currin_hf(X)
        lf = self._currin_lf(X)
        return torch.stack([hf, lf], dim=-1)  # (N, 2)

#Multi-Fidelity 1D Forrester Function:
class MultiFidelityForrester(MultiObjectiveTestProblem):
    dim = 1
    continuous_inds = [0]
    num_objectives = 3  # HF, MF, LF
    _bounds = [(0.0, 1.0)]
    _ref_point = [0.0, 0.0]  # Set something meaningful later!

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
            dtype: The dtype that is used for the bounds of the function.
        """
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    @staticmethod
    def _forrester_hf(X: torch.Tensor) -> torch.Tensor:
        x = X[..., 0]
        return (6 * x - 2)**2 * torch.sin(12 * x - 4)

    def _forrester_lf(self, X: torch.Tensor) -> torch.Tensor:
        hf = self._forrester_hf(X)
        x = X[..., 0]
        A,B,C=0.5,10,-5
        return A * hf + B * (x - 0.5) - C

    def _forrester_mf(self, X: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        hf = self._forrester_hf(X)
        lf = self._forrester_lf(X)
        return alpha * hf + (1 - alpha) * lf

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        hf = self._forrester_hf(X)
        mf = self._forrester_mf(X, alpha=0.5)  # medium fidelity
        lf = self._forrester_lf(X)
        return torch.stack([hf, mf, lf], dim=-1)  # (..., 3)

"""
### TESTING ###


mf_currin = MultiFidelityCurrin()

# Grid for plotting
n_points = 100
x = torch.linspace(0, 1, n_points)
y = torch.linspace(0, 1, n_points)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Flatten grid and evaluate using the MF class
grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)
Z = mf_currin._currin_lf(grid).reshape(n_points, n_points)  # HF only

# Plot contours
plt.figure(figsize=(6,5))
cp = plt.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=30, cmap=cm.viridis)
plt.colorbar(cp)
plt.title("High-Fidelity Currin Function (from MultiFidelityCurrin)")
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()

import matplotlib.pyplot as plt
from matplotlib import cm
mf_forrester = MultiFidelityForrester(negate=True)

# Sample some points
X = torch.linspace(0, 1, 50).unsqueeze(-1)
Y = mf_forrester(X)

plt.figure(figsize=(6,4))
plt.plot(X.numpy(), Y[:,0].numpy(), label="High-Fidelity", linewidth=2)
plt.plot(X.numpy(), Y[:,2].numpy(), label="Low-Fidelity", linewidth=2,color="blue", linestyle="--")
plt.plot(X.numpy(), Y[:,1].numpy(), label="Medium-Fidelity", linewidth=2,color="red", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("1D Forrester Function (HF vs LF)")
plt.legend()
plt.show()
#"""