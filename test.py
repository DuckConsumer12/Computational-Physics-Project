import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Equations of motion with damping
def equations(t, y):
    θ1, z1, θ2, z2 = y
    delta = θ2 - θ1

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1

    dθ1 = z1
    dθ2 = z2

    dz1 = (
        m2 * L1 * z1**2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(θ2) * np.cos(delta)
        + m2 * L2 * z2**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(θ1)
        - b * z1  # damping
    ) / den1

    dz2 = (
        -m2 * L2 * z2**2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2) * g * np.sin(θ1) * np.cos(delta)
        - (m1 + m2) * L1 * z1**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(θ2)
        - b * z2  # damping
    ) / den2

    return [dθ1, dz1, dθ2, dz2]

# Constants
g = 9.81        # gravity (m/s^2)
L1, L2 = 1.0, 0.5  # lengths of the rods (meters)
m1, m2 = 1.0, 1.0  # masses (kg)
b = 0.05      # damping coefficient (air resistance)

# Starting point (initial angles and speeds)
start_angle_1 = np.radians(35)      # starting angle of first pendulum
start_speed_1 = 0                   # starting angular speed of first pendulum
start_angle_2 = np.radians(0)      # starting angle of second pendulum
start_speed_2 = 0                   # starting angular speed of second pendulum

# Combine into initial condition list
y0 = [start_angle_1, start_speed_1, start_angle_2, start_speed_2]

# Time settings
t_span = (0, 50)
t_eval = np.linspace(*t_span, 2000)

# Solve ODE
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval)

# Plot angles over time
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='θ1 (First pendulum arm)')
plt.plot(sol.t, sol.y[2], label='θ2 (Second pendulum arm)')
plt.title("Double Pendulum with Air Resistance")
plt.xlabel("Time (s)")
plt.ylabel("Angle (radians)")
plt.legend()
plt.grid(True)
plt.tight_layout()

θ1 = sol.y[0]
θ2 = sol.y[2]
x1 = L1 * np.sin(θ1)
y1 = -L1 * np.cos(θ1)
x2 = x1 + L2 * np.sin(θ2)
y2 = y1 - L2 * np.cos(θ2)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1*(L1+L2), (L1+L2))
ax.set_ylim(-1*(L1+L2), (L1+L2))
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2)

def update(i):
  line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
  return line,

ani = FuncAnimation(fig, update, frames=len(sol.t), interval=20, blit=True)
plt.show()