import numpy as np
import matplotlib.pyplot as plt

# Create data for your 3D surface
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the viewing angle (adjust these values as needed)
ax.view_init(elev=20, azim=45)  # Change elev and azim to set the view angle

# Plot the 3D surface
surface = ax.plot_surface(x, y, z, cmap='viridis')

# Add color bar to show values
fig.colorbar(surface, shrink=0.5, aspect=5)

# Label axes (optional)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Show the plot
plt.show()
