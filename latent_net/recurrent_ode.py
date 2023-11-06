import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the ODE parameters
k = 0.1  # Decay constant

# Define the initial condition
y0 = 1.0

# Create synthetic data
num_samples = 100
dt = 0.1
t = torch.arange(0, num_samples * dt, dt)
true_solution = y0 * torch.exp(-k * t)

# Define an RNN model to approximate the ODE
class ODE_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ODE_RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input_data, hidden_state):
        output, hidden_state = self.rnn(input_data, hidden_state)
        y_pred = self.linear(output)
        return y_pred, hidden_state

# Create the RNN model
input_size = 1  # Input dimension (dy/dt)
hidden_size = 32  # Hidden state dimension
model = ODE_RNN(input_size, hidden_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
hidden_state = None  # Initial hidden state
predicted_solution = []  # List to store predicted values
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Construct the input sequence (dy/dt) and target sequence (true_solution)
    dy_dt = torch.cat([true_solution[:1], true_solution[1:] - true_solution[:-1]])
    dy_dt = dy_dt.view(1, num_samples, -1)  # Batch size of 1
    target = true_solution.view(1, num_samples, -1)

    # Forward pass
    y_pred, hidden_state = model(dy_dt, hidden_state)

    # Compute the loss
    loss = criterion(y_pred, target)

    # Backpropagation
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Append the predicted values to the list
    predicted_solution.append(y_pred.view(-1).detach().numpy())

# Convert the predicted_solution list to a NumPy array for plotting
predicted_solution = np.concatenate(predicted_solution)

# Visualize the results and compare to the true solution
import matplotlib.pyplot as plt

plt.plot(t, true_solution, label='True Solution', linewidth=2)
plt.plot(t, predicted_solution, label='Predicted Solution', linestyle='dashed')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.show()
