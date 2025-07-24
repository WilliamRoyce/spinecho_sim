from __future__ import annotations

import numpy as np

from spinecho_sim.state import CoherentSpin, Spin

if __name__ == "__main__":
    # A spin in a classical coherent state needs only two parameters,
    # theta, phi to define its orientation.
    classical_spin = CoherentSpin(theta=np.pi / 2, phi=0)
    assert classical_spin.theta == np.pi / 2
    assert classical_spin.phi == 0

    # These angles define the Cartesian coordinates of the spin vector.
    assert np.sin(classical_spin.theta) * np.cos(classical_spin.phi) == classical_spin.x
    assert np.sin(classical_spin.theta) * np.sin(classical_spin.phi) == classical_spin.y
    assert np.cos(classical_spin.theta) == classical_spin.z

    # We can also build a coherent spin from a list of (x,y,z) components.
    classical_spin_from_cartesian = CoherentSpin.from_cartesian(
        classical_spin.x.item(), classical_spin.y.item(), classical_spin.z.item()
    )
    assert classical_spin_from_cartesian.theta == classical_spin.theta
    assert classical_spin_from_cartesian.phi == classical_spin.phi
    assert classical_spin_from_cartesian == classical_spin

    # To repersent a general spin, we need to store a list of majhorana spin components.
    # We can convert between a coherent spin and a majorana spin representation
    # using the as_generic method.
    # Here, n_stars sets the number of majorana spins in the representation
    n_stars = 5
    generic_spin = classical_spin.as_generic(n_stars=n_stars)

    # We can also use a momentum state representation to analyze spin states
    # For spin 1/2, state[0] represents the spin up component and
    # state[1] represents the spin down component.
    momentum_state = generic_spin.momentum_states
    assert momentum_state.shape == (n_stars + 1,)
    # We can also build up a generic spin directly from the momentum components
    generic_from_momentum = Spin.from_momentum_state(momentum_state)
    np.testing.assert_allclose(
        generic_from_momentum.momentum_states, generic_spin.momentum_states
    )
