import torch
import numpy as np
from typing import NamedTuple, Optional, Tuple
from norse.torch.functional.threshold import threshold


class DiehlLIFState(NamedTuple):
    z: torch.Tensor
    v: torch.Tensor
    g_e: torch.Tensor
    g_i: torch.Tensor
    delta_v_thresh: torch.Tensor


# default_bio_state = DiehlLIFState(z=0.0, v=-65.0, g_e=0.0, g_i=0.0, delta_v_thresh=)


class DiehlLIFParameters(NamedTuple):
    tau_syn_exc_inv: torch.Tensor = torch.as_tensor(1.0 / 5)
    tau_syn_inh_inv: torch.Tensor = torch.as_tensor(1.0 / 5)
    tau_v_thresh_inv: torch.Tensor = torch.as_tensor(1.0 / 1e4)
    c_m_inv: torch.Tensor = torch.as_tensor(1 / 0.2)
    g_l: torch.Tensor = torch.as_tensor(1 / 20 * 1 / 0.2)
    e_rev_I: torch.Tensor = torch.as_tensor(-100)
    e_rev_E: torch.Tensor = torch.as_tensor(60)
    v_rest: torch.Tensor = torch.as_tensor(-20)
    v_reset: torch.Tensor = torch.as_tensor(-70)
    v_thresh: torch.Tensor = torch.as_tensor(-10)
    v_thresh_plus: torch.Tensor = torch.as_tensor(0.05)
    v_thresh_max: torch.Tensor = torch.as_tensor(35.0)
    method: str = "super"
    alpha: float = 100.0


# default_bio_parameters = DiehlLIFParameters(
#     tau_syn_exc_inv=1 / 0.3,
#     tau_syn_inh_inv=1 / 0.5,
#     tau_v_thresh_inv=1 / 1e4,
#     e_rev_E=0.0,
#     e_rev_I=-70.0,
#     v_thresh=-50.0,
#     v_reset=-65.0,
#     v_rest=-65.0,
#     v_thresh_plus=0.05,
#     v_thresh_max=35,
# )


def diehl_lif_step(
    input_spikes: torch.Tensor,
    state: DiehlLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: DiehlLIFParameters = DiehlLIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DiehlLIFState]:
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

    v_thresh = p.v_thresh + state.delta_v_thresh
    z_new = threshold(v - v_thresh, p.method, p.alpha)
    v = (1 - z_new) * v + z_new * p.v_reset

    delta_v_thresh = (
        1 - dt * p.tau_v_thresh_inv
    ) * state.delta_v_thresh + p.v_thresh_plus * z_new
    delta_v_thresh = torch.clamp(delta_v_thresh, 0.0, p.v_thresh_max)
    return z_new, DiehlLIFState(z_new, v, g_e, g_i, delta_v_thresh)


class DiehlLIFFeedForwardState(NamedTuple):
    v: torch.Tensor
    g_e: torch.Tensor
    g_i: torch.Tensor
    delta_v_thresh: torch.Tensor


def diehl_lif_feed_forward_step(
    input_tensor: torch.Tensor,
    state: DiehlLIFFeedForwardState,
    p: DiehlLIFParameters = DiehlLIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DiehlLIFFeedForwardState]:
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

    v_thresh = p.v_thresh + state.delta_v_thresh
    z_new = threshold(v - v_thresh, p.method, p.alpha)
    v = (1 - z_new) * v + z_new * p.v_reset

    delta_v_thresh = (
        1 - dt * p.tau_v_thresh_inv
    ) * state.delta_v_thresh + p.v_thresh_plus * z_new
    delta_v_thresh = torch.clamp(delta_v_thresh, 0.0, p.v_thresh_max)
    return z_new, DiehlLIFFeedForwardState(v, g_e, g_i, delta_v_thresh)


class DiehlLIFCell(torch.nn.Module):
    def __init__(
        self,
        p: DiehlLIFParameters = DiehlLIFParameters(),
        dt: float = 0.001,
    ):
        super(DiehlLIFCell, self).__init__()
        self.p = p
        self.dt = dt

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[DiehlLIFState] = None
    ) -> Tuple[torch.Tensor, DiehlLIFState]:
        if state is None:
            state = DiehlLIFFeedForwardState(
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
                delta_v_thresh=torch.zeros(
                    input_tensor.shape,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
            state.v.requires_grad = True
        return diehl_lif_feed_forward_step(
            input_tensor,
            state,
            p=self.p,
            dt=self.dt,
        )


class DiehlLIFRecurrentCell(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: DiehlLIFParameters = DiehlLIFParameters(),
        dt: float = 0.001,
    ):
        super(DiehlLIFRecurrentCell, self).__init__()
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
        self, input_tensor: torch.Tensor, state: Optional[DiehlLIFState] = None
    ) -> Tuple[torch.Tensor, DiehlLIFState]:
        if state is None:
            state = DiehlLIFState(
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
                delta_v_thresh=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
            state.v.requires_grad = True
        return diehl_lif_step(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            p=self.p,
            dt=self.dt,
        )
