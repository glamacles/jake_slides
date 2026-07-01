"""Standalone UDE glacier example (non-notebook version).

Recovers Glen's creep parameter A(T) from synthetic surface-velocity
observations by training a neural network *inside* the 1-D shallow-ice
flowline equation and differentiating through the solver.

This is the same, validated experiment that powers `UDE_glaciology_lecture.ipynb`
-- kept as a plain script for quick runs / debugging outside Jupyter.

Run:  uv run python prototype.py     (~30 s on a CPU)
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
import diffrax as dfx
import equinox as eqx
import optax
import optimistix as optx

jax.config.update("jax_enable_x64", True)  # glacier dynamics are stiff -> need f64

# --- physical constants (SI, time in years) -------------------------------
RHO, G = 900.0, 9.81
N_GLEN = 3.0
A_FLOOR, A_CEIL = 1e-18, 4e-16            # solver-safe band for the creep param

# --- flowline geometry & climate ------------------------------------------
L, NX = 30_000.0, 41
dx = L / (NX - 1)
x = jnp.linspace(0.0, L, NX)
bed = 2800.0 - 0.04 * x
ELA, BETA, CAP = 2300.0, 0.004, 0.8


def mass_balance(S):
    return jnp.clip(BETA * (S - ELA), None, CAP)


def sia_rhs(t, H, A):
    """dH/dt for the SIA flowline. `A` is the term we later learn."""
    A = jnp.clip(A, A_FLOOR, A_CEIL)
    H = jnp.clip(H, 0.0)
    S = bed + H
    dSdx = (S[1:] - S[:-1]) / dx
    H_edge = 0.5 * (H[1:] + H[:-1])
    Gamma = 2.0 * A / (N_GLEN + 2.0) * (RHO * G) ** N_GLEN
    D = Gamma * H_edge ** (N_GLEN + 2.0) * jnp.abs(dSdx) ** (N_GLEN - 1.0)
    q = -D * dSdx
    div = jnp.zeros(NX).at[1:-1].set(-(q[1:] - q[:-1]) / dx)
    bdot = mass_balance(S)
    limiter = jnp.clip(H / 10.0, 0.0, 1.0)          # smooth no-melt-without-ice
    bdot_eff = jnp.maximum(bdot, 0.0) + jnp.minimum(bdot, 0.0) * limiter
    return bdot_eff + div


# Implicit backward-Euler: ice flow is stiff diffusion, and an implicit solver
# gives well-behaved gradients (an explicit solver's gradients explode on stiff
# ice -- see numerical_deep_dive.ipynb). Each step solves a nonlinear system.
SOLVER = dfx.ImplicitEuler(root_finder=dfx.VeryChord(rtol=1e-5, atol=1e-4, norm=optx.max_norm))
DT_YEARS = 0.5


def solve_sia(A, H0, t1):
    n = int(round(t1 / DT_YEARS))
    sol = dfx.diffeqsolve(
        dfx.ODETerm(sia_rhs), SOLVER,
        t0=0.0, t1=t1, dt0=DT_YEARS, y0=H0, args=A,
        stepsize_controller=dfx.ConstantStepSize(), max_steps=n + 5,
        saveat=dfx.SaveAt(t1=True),
        adjoint=dfx.RecursiveCheckpointAdjoint(),
    )
    return sol.ys[-1]


def surface_velocity(H, A):
    A = jnp.clip(A, A_FLOOR, A_CEIL)
    H = jnp.clip(H, 0.0)
    S = bed + H
    dSdx = (S[1:] - S[:-1]) / dx
    H_edge = 0.5 * (H[1:] + H[:-1])
    u_edge = (2.0 * A / (N_GLEN + 1.0)) * (RHO * G) ** N_GLEN \
        * H_edge ** (N_GLEN + 1.0) * jnp.abs(dSdx) ** N_GLEN
    return jnp.zeros(NX).at[1:-1].set(0.5 * (u_edge[1:] + u_edge[:-1]))


# --- hidden truth & synthetic data ----------------------------------------
T_glaciers = jnp.linspace(-25.0, -3.0, 8)


def A_true(T):
    return 1.0e-16 * jnp.exp(0.13 * (T + 10.0))


A_REF = float(A_true(-14.0))
T_SPIN, T_OBS = 300.0, 30.0


class CreepNet(eqx.Module):
    """Learns log(A): output A_ref * exp(net(T)) -> scale-free, positive."""
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(1, 1, width_size=10, depth=2,
                              activation=jax.nn.softplus, key=key)

    def __call__(self, T):
        z = jnp.atleast_1d((T + 14.0) / 11.0)
        raw = self.mlp(z)[0]
        return A_REF * jnp.exp(jnp.clip(raw, -5.0, 5.0))


def loss_fn(model, U_obs, H_init):
    A_pred = jax.vmap(model)(T_glaciers)
    H_pred = jax.vmap(lambda A: solve_sia(A, H_init, T_OBS))(A_pred)
    U_pred = jax.vmap(surface_velocity)(H_pred, A_pred)
    scale = jnp.maximum(U_obs.max(axis=1, keepdims=True), 1.0)
    return jnp.mean(((U_pred - U_obs) / scale) ** 2)


if __name__ == "__main__":
    H_init = solve_sia(A_REF, jnp.zeros(NX), T_SPIN)
    A_vals = A_true(T_glaciers)
    H_obs = jax.vmap(lambda A: solve_sia(A, H_init, T_OBS))(A_vals)
    U_obs = jax.vmap(surface_velocity)(H_obs, A_vals)
    print(f"[data] start geometry maxH={float(H_init.max()):.0f} m  "
          f"peak vel/glacier (m/a)={np.round(np.array(U_obs.max(1)),1)}")

    N_EPOCHS = 400
    sched = optax.cosine_decay_schedule(3e-3, N_EPOCHS, alpha=0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    model = CreepNet(jax.random.PRNGKey(0))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, U_obs, H_init):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, U_obs, H_init)
        updates, opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    t0 = time.time()
    for it in range(N_EPOCHS):
        model, opt_state, loss = step(model, opt_state, U_obs, H_init)
        if it % 50 == 0 or it == N_EPOCHS - 1:
            print(f"  epoch {it:4d}  loss={float(loss):.4e}")
    print(f"[train] {N_EPOCHS} epochs in {time.time()-t0:.1f}s")

    A_learned = jax.vmap(model)(T_glaciers)
    rel = np.abs(np.array((A_learned - A_vals) / A_vals))
    print(f"[recovery] A(T) mean rel. error {rel.mean():.1%}  max {rel.max():.1%}")
