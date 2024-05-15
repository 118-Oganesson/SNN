from typing import NamedTuple, Tuple
import torch
import torch.jit

from norse.torch.functional.threshold import threshold
import norse.torch.utils.pytree as pytree
from norse.torch.module.snn import SNNCell, SNNRecurrentCell
from norse.torch.utils.clone import clone_tensor


class DiehlLIFParameters(
    pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta
):
    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    tau_v_th_inv: torch.Tensor = torch.as_tensor(1.0 / 1e4)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    v_th_plus: torch.Tensor = torch.as_tensor(0.05)
    v_th_max: torch.Tensor = torch.as_tensor(35.0)
    method: str = "super"
    alpha: float = torch.as_tensor(100.0)


# pytype: disable=bad-unpacking,wrong-keyword-args
# default_bio_parameters = DiehlLIFParameters(
#     tau_syn_inv=torch.as_tensor(1 / 0.5),
#     tau_mem_inv=torch.as_tensor(1 / 20.0),
#     tau_v_th_inv=torch.as_tensor(1.0 / 1e4),
#     v_leak=torch.as_tensor(-65.0),
#     v_th=torch.as_tensor(-50.0),
#     v_reset=torch.as_tensor(-65.0),
#     v_th_plus=torch.as_tensor(0.05),
#     v_th_max=torch.as_tensor(35.0),
# )
# pytype: enable=bad-unpacking,wrong-keyword-args


class DiehlLIFState(NamedTuple):
    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor
    dv_th: torch.Tensor


# default_bio_initial_state = DiehlLIFState(
#     z=torch.as_tensor(0.0), v=torch.as_tensor(-65.0), i=torch.as_tensor(0.0)
# )


class DiehlLIFFeedForwardState(NamedTuple):
    v: torch.Tensor
    i: torch.Tensor
    dv_th: torch.Tensor


def diehl_lif_step_sparse(
    input_spikes: torch.Tensor,
    state: DiehlLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: DiehlLIFParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DiehlLIFState]:
    # compute current jumps
    i_jump = (
        state.i
        + torch.sparse.mm(input_spikes, input_weights.t())
        + torch.sparse.mm(state.z, recurrent_weights.t())
    )

    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    # compute new spikes
    v_th = p.v_th + state.dv_th
    z_new = threshold(v_decayed - v_th, p.method, p.alpha)

    # compute dv_th update
    dv_th = (1 - dt * p.tau_v_th_inv) * state.dv_th + z_new * p.v_th_plus
    dv_th = torch.clamp(dv_th, 0.0, p.v_th_max)

    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

    z_sparse = z_new.to_sparse()
    return z_sparse, DiehlLIFState(z_sparse, v_new, i_decayed, dv_th)


def diehl_lif_step(
    input_spikes: torch.Tensor,
    state: DiehlLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: DiehlLIFParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DiehlLIFState]:
    # compute current jumps
    i_jump = (
        state.i
        + torch.nn.functional.linear(input_spikes, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    # compute new spikes
    v_th = p.v_th + state.dv_th
    z_new = threshold(v_decayed - v_th, p.method, p.alpha)

    # compute dv_th update
    dv_th = (1 - dt * p.tau_v_th_inv) * state.dv_th + z_new * p.v_th_plus
    dv_th = torch.clamp(dv_th, 0.0, p.v_th_max)

    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

    return z_new, DiehlLIFState(z_new, v_new, i_decayed, dv_th)


def diehl_lif_feed_forward_step(
    input_spikes: torch.Tensor,
    state: DiehlLIFFeedForwardState,
    p: DiehlLIFParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DiehlLIFFeedForwardState]:
    # compute current jumps
    i_new = state.i + input_spikes

    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_new)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_new
    i_decayed = i_new + di

    # compute new spikes
    v_th = p.v_th + state.dv_th
    z_new = threshold(v_decayed - v_th, p.method, p.alpha)

    # compute dv_th update
    dv_th = (1 - dt * p.tau_v_th_inv) * state.dv_th + z_new * p.v_th_plus
    dv_th = torch.clamp(dv_th, 0.0, p.v_th_max)

    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

    return z_new, DiehlLIFFeedForwardState(v_new, i_decayed, dv_th)


def diehl_lif_feed_forward_step_sparse(
    input_tensor: torch.Tensor,
    state: DiehlLIFFeedForwardState,
    p: DiehlLIFParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DiehlLIFFeedForwardState]:  # pragma: no cover
    # compute current jumps
    i_jump = state.i + input_tensor

    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    # compute new spikes
    v_th = p.v_th + state.dv_th
    z_new = threshold(v_decayed - v_th, p.method, p.alpha)

    # compute dv_th update
    dv_th = (1 - dt * p.tau_v_th_inv) * state.dv_th + z_new * p.v_th_plus
    dv_th = torch.clamp(dv_th, 0.0, p.v_th_max)

    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

    return z_new.to_sparse(), DiehlLIFFeedForwardState(v_new, i_decayed, dv_th)


class DiehlLIFCell(SNNCell):
    def __init__(self, p: DiehlLIFParameters = DiehlLIFParameters(), **kwargs):
        super().__init__(
            # activation=(
            #     diehl_lif_feed_forward_adjoint_step
            #     if p.method == "adjoint"
            #     else diehl_lif_feed_forward_step
            # ),
            # activation_sparse=(
            #     diehl_lif_feed_forward_adjoint_step_sparse
            #     if p.method == "adjoint"
            #     else diehl_lif_feed_forward_step_sparse
            # ),
            activation=diehl_lif_feed_forward_step,
            activation_sparse=diehl_lif_feed_forward_step_sparse,
            state_fallback=self.initial_state,
            p=DiehlLIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.tau_v_th_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                torch.as_tensor(p.v_th_plus),
                torch.as_tensor(p.v_th_max),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> DiehlLIFFeedForwardState:
        state = DiehlLIFFeedForwardState(
            v=clone_tensor(self.p.v_leak),
            i=torch.zeros(
                input_tensor.shape,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            dv_th=torch.zeros(
                input_tensor.shape,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state


class DiehlLIFRecurrentCell(SNNRecurrentCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: DiehlLIFParameters = DiehlLIFParameters(),
        **kwargs,
    ):
        super().__init__(
            # activation=diehl_lif_adjoint_step
            # if p.method == "adjoint"
            # else diehl_lif_step,
            # activation_sparse=(
            #     diehl_lif_adjoint_step_sparse
            #     if p.method == "adjoint"
            #     else diehl_lif_step_sparse
            # ),
            activation=diehl_lif_step,
            activation_sparse=diehl_lif_step_sparse,
            state_fallback=self.initial_state,
            p=DiehlLIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.tau_v_th_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                torch.as_tensor(p.v_th_plus),
                torch.as_tensor(p.v_th_max),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            input_size=input_size,
            hidden_size=hidden_size,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> DiehlLIFState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        state = DiehlLIFState(
            z=(
                torch.zeros(
                    dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ).to_sparse()
                if input_tensor.is_sparse
                else torch.zeros(
                    dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                )
            ),
            v=torch.full(
                dims,
                torch.as_tensor(self.p.v_leak).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                dims,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            dv_th=torch.zeros(
                dims,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state
