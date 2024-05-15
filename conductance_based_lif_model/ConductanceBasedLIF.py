from typing import Optional, NamedTuple, Tuple

import torch

import numpy as np

from norse.torch.functional.threshold import threshold


class CobaLIFState(NamedTuple):
    z: torch.Tensor
    v: torch.Tensor
    g_e: torch.Tensor
    g_i: torch.Tensor


default_bio_state = CobaLIFState(z=0.0, v=-65.0, g_e=0.0, g_i=0.0)


class CobaLIFParameters(NamedTuple):
    tau_syn_exc_inv: torch.Tensor = torch.as_tensor(1.0 / 5)
    tau_syn_inh_inv: torch.Tensor = torch.as_tensor(1.0 / 5)
    c_m_inv: torch.Tensor = torch.as_tensor(1 / 0.2)
    g_l: torch.Tensor = torch.as_tensor(1 / 20 * 1 / 0.2)
    e_rev_I: torch.Tensor = torch.as_tensor(-100)
    e_rev_E: torch.Tensor = torch.as_tensor(60)
    v_rest: torch.Tensor = torch.as_tensor(-20)
    v_reset: torch.Tensor = torch.as_tensor(-70)
    v_thresh: torch.Tensor = torch.as_tensor(-10)
    method: str = "super"
    alpha: float = 100.0


default_bio_parameters = CobaLIFParameters(
    tau_syn_exc_inv=1 / 0.3,
    tau_syn_inh_inv=1 / 0.5,
    e_rev_E=0.0,
    e_rev_I=-70.0,
    v_thresh=-50.0,
    v_reset=-65.0,
    v_rest=-65.0,
)


def coba_lif_step(
    input_spikes: torch.Tensor,
    state: CobaLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: CobaLIFParameters = CobaLIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, CobaLIFState]:
    # conductance jumps
    g_e = state.g_e + torch.nn.functional.linear(
        input_spikes, torch.nn.functional.relu(input_weights)
    )
    g_i = state.g_i + torch.nn.functional.linear(
        input_spikes, torch.nn.functional.relu(-input_weights)
    )

    g_e += torch.nn.functional.linear(
        state.z, torch.nn.functional.relu(recurrent_weights)
    )
    g_i += torch.nn.functional.linear(
        state.z, torch.nn.functional.relu(-recurrent_weights)
    )
    dg_e = -dt * p.tau_syn_exc_inv * g_e
    g_e = g_e + dg_e
    dg_i = -dt * p.tau_syn_inh_inv * g_i
    g_i = g_i + dg_i

    dv = (
        dt
        * p.c_m_inv
        * (
            p.g_l * (p.v_rest - state.v)
            + g_e * (p.e_rev_E - state.v)
            + g_i * (p.e_rev_I - state.v)
        )
    )
    v = state.v + dv

    z_new = threshold(v - p.v_thresh, p.method, p.alpha)
    v = (1 - z_new) * v + z_new * p.v_reset
    return z_new, CobaLIFState(z_new, v, g_e, g_i)


class CobaLIFFeedForwardState(NamedTuple):
    v: torch.Tensor
    g_e: torch.Tensor
    g_i: torch.Tensor


def coba_lif_feed_forward_step(
    input_tensor: torch.Tensor,
    state: CobaLIFFeedForwardState,
    p: CobaLIFParameters = CobaLIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, CobaLIFFeedForwardState]:
    # conductance jumps
    g_e = state.g_e + torch.nn.functional.relu(input_tensor)
    g_i = state.g_i + torch.nn.functional.relu(-input_tensor)

    dg_e = -dt * p.tau_syn_exc_inv * g_e
    g_e = g_e + dg_e
    dg_i = -dt * p.tau_syn_inh_inv * g_i
    g_i = g_i + dg_i

    dv = (
        dt
        * p.c_m_inv
        * (
            p.g_l * (p.v_rest - state.v)
            + g_e * (p.e_rev_E - state.v)
            + g_i * (p.e_rev_I - state.v)
        )
    )
    v = state.v + dv

    z_new = threshold(v - p.v_thresh, p.method, p.alpha)
    v = (1 - z_new) * v + z_new * p.v_reset
    return z_new, CobaLIFFeedForwardState(v, g_e, g_i)


class CobaLIFCell(torch.nn.Module):
    def __init__(
        self,
        p: CobaLIFParameters = CobaLIFParameters(),
        dt: float = 0.001,
    ):
        super(CobaLIFCell, self).__init__()
        self.p = p
        self.dt = dt

    def forward(
        self,
        input_tensor: torch.Tensor,
        state: Optional[CobaLIFFeedForwardState] = None,
    ) -> Tuple[torch.Tensor, CobaLIFFeedForwardState]:
        if state is None:
            state = CobaLIFFeedForwardState(
                v=torch.zeros(
                    input_tensor.shape,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                g_e=torch.zeros(
                    input_tensor.shape,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                g_i=torch.zeros(
                    input_tensor.shape,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
            state.v.requires_grad = True
        return coba_lif_feed_forward_step(
            input_tensor,
            state,
            p=self.p,
            dt=self.dt,
        )


class CobaLIFRecurrentCell(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: CobaLIFParameters = CobaLIFParameters(),
        dt: float = 0.001,
    ):
        super(CobaLIFRecurrentCell, self).__init__()
        self.input_weights = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) / np.sqrt(input_size)
        )
        self.recurrent_weights = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = p
        self.dt = dt

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[CobaLIFState] = None
    ) -> Tuple[torch.Tensor, CobaLIFState]:
        if state is None:
            state = CobaLIFState(
                z=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                v=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                g_e=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                g_i=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
            state.v.requires_grad = True
        return coba_lif_step(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            p=self.p,
            dt=self.dt,
        )
