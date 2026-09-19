"""APQB state algebra -- the scalar reference semantics of the Qubit VM.

An Adjustable Pseudo Quantum Bit (APQB) is a point on the unit circle of the
complex coordinate plane of the QBNN/APQB paper (Qubit repo, apqb_qbnn_v2.py):

    z = r + i*eta = e^{i*2*theta},   r^2 + eta^2 = 1

with the three views the paper uses kept in sync:

    r     = cos(2 theta)   statistical correlation coordinate, in [-1, 1]
    eta   = sin(2 theta)   coherence / "uncertainty" coordinate (paper's q)
    theta                  the internal angle, in (-pi/2, pi/2]
    T     = |eta|          the AI-temperature analogue, |sin(2 theta)|

The paper restricts canonical states to eta >= 0 (eta = sech(a) is positive).
Products of states -- the subset features Phi_S(z) = prod_{i in S} z_i of
Eq. (17)-(18) -- leave that half circle, so the VM carries the full circle and
reports the paper's T as |eta|.  Every function here is pure and operates on
plain floats; the vector backends in qubitbridge.backends mirror them lane-wise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "EPS",
    "R_MAX",
    "clamp_atanh",
    "APQBState",
    "ENCODINGS",
    "sech",
    "clamp_unit",
    "encode",
    "decode",
    "rotate",
    "interact",
    "state_mul",
    "correlate",
    "uncertainty",
    "entropy_z",
    "gate",
    "measure_expect",
    "measure_sample",
    "power",
    "probabilities",
]

#: Distance kept from the degenerate endpoints r = +-1, where atanh diverges.
EPS = 1e-12

#: The largest double below 1.  atanh() is clamped to +-R_MAX rather than
#: shrunk by a factor, so the bound is exactly representable and every backend
#: -- Python, NumPy, AArch64 -- can reproduce it bit for bit.
R_MAX = math.nextafter(1.0, 0.0)

#: Encoding modes understood by :func:`encode` / :func:`decode`.
ENCODINGS = ("latent", "linear", "angle", "prob")


def sech(a: float) -> float:
    """sech(a) = 2 / (e^a + e^-a), evaluated without overflowing cosh."""
    m = abs(a)
    e = math.exp(-m)
    return 2.0 * e / (1.0 + e * e)


def clamp_unit(x: float) -> float:
    """Clamp into [-1, 1]."""
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)


def clamp_atanh(x: float) -> float:
    """Clamp into [-R_MAX, R_MAX], the domain where atanh stays finite."""
    return -R_MAX if x < -R_MAX else (R_MAX if x > R_MAX else x)


@dataclass(frozen=True)
class APQBState:
    """One pseudo qubit: the register contents ``{r, eta, theta}``.

    ``r`` and ``eta`` are stored; ``theta`` is derived so that the three can
    never drift apart.  Construct through :meth:`from_r`, :meth:`from_theta`,
    :meth:`from_latent` or :func:`encode` rather than calling the constructor
    with an off-circle pair.
    """

    r: float
    eta: float

    @property
    def theta(self) -> float:
        """theta = (1/2) atan2(eta, r), the paper's internal angle."""
        return 0.5 * math.atan2(self.eta, self.r)

    @property
    def T(self) -> float:
        """The AI-temperature analogue T = |sin(2 theta)| = |eta|."""
        return abs(self.eta)

    @property
    def z(self) -> complex:
        """The complex coordinate z = r + i*eta of Eq. (14)."""
        return complex(self.r, self.eta)

    def normalized(self) -> "APQBState":
        """Re-project onto the unit circle, undoing accumulated rounding."""
        n = math.hypot(self.r, self.eta)
        if n == 0.0:
            return APQBState(1.0, 0.0)
        return APQBState(self.r / n, self.eta / n)

    def constraint_error(self) -> float:
        """r^2 + eta^2 - 1; ~0 for any well formed state."""
        return self.r * self.r + self.eta * self.eta - 1.0

    # -- canonical constructors ------------------------------------------
    @staticmethod
    def from_r(r: float) -> "APQBState":
        """Canonical (eta >= 0) state with the given correlation coordinate."""
        r = clamp_unit(r)
        return APQBState(r, math.sqrt(max(0.0, 1.0 - r * r)))

    @staticmethod
    def from_theta(theta: float) -> "APQBState":
        """State at internal angle theta: r = cos(2t), eta = sin(2t)."""
        two = 2.0 * theta
        return APQBState(math.cos(two), math.sin(two))

    @staticmethod
    def from_latent(a: float) -> "APQBState":
        """Eq. (12): r = tanh(a), eta = sech(a); the invariant holds exactly."""
        return APQBState(math.tanh(a), sech(a))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"APQBState(r={self.r:.6f}, eta={self.eta:.6f}, "
                f"theta={self.theta:.6f})")


#: The state every register is initialised to: r=1, eta=0, i.e. |0>.
ZERO_STATE = APQBState(1.0, 0.0)


def encode(x: float, mode: str = "latent") -> APQBState:
    """Map a classical scalar onto the APQB circle.

    ``latent``  unconstrained a  -> r = tanh(a), eta = sech(a)   (Eq. 12)
    ``linear``  x already in [-1, 1] -> r = x                    (r = cos 2t)
    ``angle``   x is theta itself
    ``prob``    x in [0, 1] is P(0) -> r = 2x - 1                (Eq. 6-7)
    """
    if mode == "latent":
        return APQBState.from_latent(x)
    if mode == "linear":
        return APQBState.from_r(x)
    if mode == "angle":
        return APQBState.from_theta(x)
    if mode == "prob":
        return APQBState.from_r(2.0 * x - 1.0)
    raise ValueError(f"unknown APQB encoding {mode!r}, expected one of {ENCODINGS}")


def decode(s: APQBState, mode: str = "latent") -> float:
    """Inverse of :func:`encode` on the r coordinate."""
    if mode == "latent":
        return math.atanh(clamp_atanh(s.r))
    if mode == "linear":
        return s.r
    if mode == "angle":
        return s.theta
    if mode == "prob":
        return 0.5 * (1.0 + s.r)
    raise ValueError(f"unknown APQB encoding {mode!r}, expected one of {ENCODINGS}")


def rotate(s: APQBState, phi: float) -> APQBState:
    """QROT: theta -> theta + phi, i.e. z -> z * e^{i*2*phi}."""
    two = 2.0 * phi
    c, sn = math.cos(two), math.sin(two)
    return APQBState(s.r * c - s.eta * sn, s.r * sn + s.eta * c)


def interact(a: APQBState, b: APQBState) -> APQBState:
    """QINT: the subset product z_a * z_b of Eq. (17), i.e. theta addition.

    This is the VM's fundamental two-body operation: repeating it builds the
    degree-k terms z^k whose real and imaginary parts are the Chebyshev
    features T_k(r) and eta*U_{k-1}(r) of Prop. 2.
    """
    return APQBState(a.r * b.r - a.eta * b.eta, a.r * b.eta + a.eta * b.r)


def state_mul(a: APQBState, b: APQBState) -> APQBState:
    """QMUL: product of the correlation coordinates, r = r_a * r_b.

    Unlike :func:`interact` this stays on the canonical half circle: [-1, 1]
    is closed under multiplication, so the result is a valid correlation and
    eta is re-derived as +sqrt(1 - r^2).
    """
    return APQBState.from_r(a.r * b.r)


def correlate(a: APQBState, b: APQBState) -> float:
    """QCORR: Re(z_a * conj(z_b)) = cos(2(theta_a - theta_b))."""
    return a.r * b.r + a.eta * b.eta


def uncertainty(s: APQBState) -> float:
    """QUNC: T = |eta| = |sin(2 theta)|, the l1 coherence of Sec. 3.4."""
    return abs(s.eta)


def entropy_z(s: APQBState) -> float:
    """H_Z, the Shannon entropy in bits of a Z-basis measurement (Eq. 10)."""
    p0, p1 = probabilities(s)
    out = 0.0
    for p in (p0, p1):
        if p > 0.0:
            out -= p * math.log2(p)
    return out


def gate(target: APQBState, source: APQBState, j: float) -> APQBState:
    """QGATE: the QBNN coupling ``J`` applied in latent space.

        a  = atanh(r_target) + j * r_source
        r' = tanh(a),  eta' = sech(a)

    Working through the latent coordinate keeps the result inside [-1, 1] for
    any coupling strength, and j = 0 is exactly the identity -- the discrete
    counterpart of the ``lambda = 0`` reduction of QBNNLayerV2.
    """
    if j == 0.0:
        return target
    a = math.atanh(clamp_atanh(target.r)) + j * source.r
    return APQBState.from_latent(a)


def probabilities(s: APQBState) -> tuple[float, float]:
    """Eq. (6)-(7): P(0) = (1 + r)/2, P(1) = (1 - r)/2."""
    r = clamp_unit(s.r)
    return 0.5 * (1.0 + r), 0.5 * (1.0 - r)


def measure_expect(s: APQBState) -> float:
    """QMEASURE in ``expect`` mode: the deterministic expectation <Z> = r."""
    return s.r


def measure_sample(s: APQBState, u: float) -> float:
    """QMEASURE in ``sample`` mode: +1 / -1 drawn with P(0) = (1 + r)/2.

    ``u`` is a uniform draw in [0, 1) supplied by the VM, so a seeded run is
    bit-for-bit reproducible across every backend.
    """
    p0, _ = probabilities(s)
    return 1.0 if u < p0 else -1.0


def power(s: APQBState, k: int) -> APQBState:
    """QPOW: z^k by repeated :func:`interact` (k >= 0); z^0 = |0>."""
    if k < 0:
        raise ValueError("APQB power requires k >= 0")
    acc = ZERO_STATE
    for _ in range(k):
        acc = interact(acc, s)
    return acc
