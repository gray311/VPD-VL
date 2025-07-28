import torch
import torch.optim as optim
import matplotlib.pyplot as plt


# Define the bicycle model
class BicycleModel:
    def __init__(self, wheelbase=2.0):
        self.wheelbase = wheelbase

    def step(self, state, control, dt):
        x, y, v, h = state
        a, delta = control

        x_next = x + v * torch.cos(h) * dt
        y_next = y + v * torch.sin(h) * dt
        v_next = v + a * dt
        h_next = h + (v / self.wheelbase) * torch.tan(delta) * dt

        # Use torch.stack to combine results into a tensor while preserving the computation graph
        return torch.stack([x_next, y_next, v_next, h_next])


# Function to vectorize initialization of controls using inverse dynamics
def initialize_controls(states, target_states, dt, wheelbase):
    states = torch.tensor(states)
    target_states = torch.tensor(target_states)

    v_current = states[:, 2]
    v_target = target_states[:, 2]
    a = (v_target - v_current) / dt

    h_current = states[:, 3]
    h_target = target_states[:, 3]

    epsilon = 1e-5
    v_nonzero = torch.where(v_current == 0, epsilon, v_current)
    delta = torch.atan2((h_target - h_current), (v_nonzero / wheelbase))
    controls = torch.stack((a, delta), dim=1)

    return controls


# Define the loss function for trajectory tracking
def trajectory_loss(predicted_trajectory, target_trajectory):
    return torch.mean((predicted_trajectory - target_trajectory) ** 2)


# Plotting function
def plot_trajectories(reference_trajectory, optimized_trajectory):
    plt.figure(figsize=(10, 6))
    plt.plot(
        reference_trajectory[:, 0],
        reference_trajectory[:, 1],
        "r--",
        label="Reference Trajectory",
    )
    plt.plot(
        optimized_trajectory[:, 0],
        optimized_trajectory[:, 1],
        "b-",
        label="Optimized Trajectory",
    )
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("trajectory_comparison.png")


# Initial state and complex target trajectory
initial_state = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
target_trajectory = torch.tensor(
    [[1.0, 2.0, 1.0, 0.1], [2.5, 3.5, 1.5, 0.3], [4.0, 5.0, 2.0, 0.5]]
)

# Prepare states for initialization
dt = 0.1
controls = torch.zeros((target_trajectory.size(0), 2), requires_grad=True)

# Optimization setup
optimizer = optim.Adam([controls], lr=0.1)
model = BicycleModel()

# Optimization loop
for epoch in range(100):
    optimizer.zero_grad()

    # Simulate trajectory
    state = initial_state.clone()
    predicted_trajectory = []

    for t in range(target_trajectory.size(0)):
        state = model.step(state, controls[t], dt=dt)
        predicted_trajectory.append(state)

    predicted_trajectory = torch.stack(predicted_trajectory)

    # Compute loss and update controls
    loss = trajectory_loss(predicted_trajectory[:, :2], target_trajectory[:, :2])

    print(loss)
    loss.backward()
    optimizer.step()

# Plot the results
plot_trajectories(
    target_trajectory.detach().numpy(), predicted_trajectory.detach().numpy()
)
