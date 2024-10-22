import os
import sys

import jax.numpy as jnp

# Add 'src/cbfkit' to the Python path
script_dir = os.path.dirname(os.path.realpath(__file__))
cbfkit_path = os.path.join(script_dir, '..', '..', 'src')
sys.path.append(cbfkit_path)
print(script_dir, "------", cbfkit_path)

from cbfkit.sensors import unbiased_gaussian_noise_sd as noisy_sensor

x = jnp.array([1.0, 1.0, 1.0, 1.0])
t = 0.0

result = noisy_sensor(t, x)

print(result)



