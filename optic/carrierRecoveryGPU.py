from numba import cuda
import cupy as cp
from cupyx import jit

def bps(Ei, N, constSymb, B):
    """
    Blind phase search (BPS) algorithm

    Parameters
    ----------
    Ei : complex-valued ndarray
        Received constellation symbols.
    N : int
        Half of the 2*N+1 average window.
    constSymb : complex-valued ndarray
        Complex-valued constellation.
    B : int
        number of test phases.

    Returns
    -------
    θ : real-valued ndarray
        Time-varying estimated phase-shifts.

    """
    nModes = Ei.shape[1]
    num_constellation_symbols = constSymb.shape[0]

    Ei = cp.asarray(Ei)
    constSymb = cp.asarray(constSymb)

    ϕ_test = cp.arange(0, B) * (cp.pi / 2) / B  # test phases

    θ = cp.zeros(Ei.shape, dtype="float")

    zeroPad = cp.zeros((N, nModes), dtype="complex")
    x = cp.concatenate(
        (zeroPad, Ei, zeroPad)
    )  # pad start and end of the signal with zeros

    L = x.shape[0]

    for n in range(0, nModes):

        dist = cp.zeros((B, num_constellation_symbols), dtype="float")
        dmin = cp.zeros((B, 2 * N + 1), dtype="float")

        for k in range(0, L):
            for indPhase, ϕ in enumerate(ϕ_test):
                dist[indPhase, :] = cp.abs(x[k, n] * cp.exp(1j * ϕ) - constSymb) ** 2
                dmin[indPhase, -1] = cp.min(dist[indPhase, :])
            if k >= 2 * N:
                sumDmin = cp.sum(dmin, axis=1)
                indRot = cp.argmin(sumDmin)
                θ[k - 2 * N, n] = ϕ_test[indRot]
            dmin = cp.roll(dmin, -1)
    return cp.asnumpy(θ)
