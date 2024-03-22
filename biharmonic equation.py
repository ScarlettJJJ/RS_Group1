import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Create a list of N
N_lis = [20, 50, 80, 100, 150, 200, 300, 500, 1000]

# Define function f(y) and g(y)
def f(z, C = 30, lamb = 2/10):
    # Step function
    if 1/2 - lamb/2 < z < 1/2 + lamb/2:
        return C * ((lamb/2)**2 - (z-1/2)**2)
    else:
        return 0

def g(z):
    return f(z)  # Since g(y) = f(y) in the problem definition

# Function to map 2D grid coordinates to 1D vector index
def idx(i, j):
    return i * (Ny+1) + j

# Loop through the list of N
for Nx in N_lis:
    Ny = Nx
    x = np.linspace(0, 1, Nx + 1)
    y = np.linspace(0, 1, Ny + 1)
    dx, dy = x[1] - x[0], y[1] - y[0]

    # Count the number of rows in the matrix A
    count_row = 0

    # Initialize the discretization matrix and right-hand side vector
    A = np.zeros(((Nx + 1) * (Ny + 1),
                  (Nx + 1) * (Ny + 1)))
    b = np.zeros((Nx + 1) * (Ny + 1))  # the right-hand side vector, len(b) = the number of rows in A

    # Fill the matrix A and vector b for the internal points
    for i in range(2, Nx - 1):  # Start from 2 and end at Nx-2 to avoid out-of-bounds
        for j in range(2, Ny - 1):  # Start from 2 and end at Ny-2 for the same reason
            index = idx(i, j)
            # Ui,j-2 + 2Ui-1,j-1 - 8Ui,j-1 + 2Ui+1,j-1 + Ui-2,j - 8Ui-1,j + 20Ui,j - 8Ui+1,j + Ui+2,j + 2Ui-1,j+1 - 8Ui,j+1 + 2Ui+1,j+1 + Ui,j+2 = 0
            A[count_row, idx(i, j - 2)] = 1
            A[count_row, idx(i - 1, j - 1)] = 2
            A[count_row, idx(i, j - 1)] = -8
            A[count_row, idx(i + 1, j - 1)] = 2
            A[count_row, idx(i - 2, j)] = 1
            A[count_row, idx(i - 1, j)] = -8
            A[count_row, idx(i, j)] = 20
            A[count_row, idx(i + 1, j)] = -8
            A[count_row, idx(i + 2, j)] = 1
            A[count_row, idx(i - 1, j + 1)] = 2
            A[count_row, idx(i, j + 1)] = -8
            A[count_row, idx(i + 1, j + 1)] = 2
            A[count_row, idx(i, j + 2)] = 1

            b[count_row] = 0
            count_row += 1

    for i in range(1, Nx):
        A[count_row, idx(i - 1, 0)] = -1
        A[count_row, idx(i + 1, 0)] = 1

        b[count_row] = 0
        count_row += 1

    for i in range(1, Nx):
        A[count_row, idx(i - 1, Ny)] = -1
        A[count_row, idx(i + 1, Ny)] = 1

        b[count_row] = 0
        count_row += 1

    for j in range(1, Ny):
        A[count_row, idx(0, j - 1)] = -1 / (2 * dy)
        A[count_row, idx(0, j + 1)] = 1 / (2 * dy)

        b[count_row] = f(j * dy)
        count_row += 1

    for j in range(1, Ny):
        A[count_row, idx(Nx, j - 1)] = -1 / (2 * dy)
        A[count_row, idx(Nx, j + 1)] = 1 / (2 * dy)

        b[count_row] = g(j * dy)
        count_row += 1

    A[count_row, idx(0, 0)] = -3 / 2
    A[count_row, idx(1, 0)] = 2
    A[count_row, idx(2, 0)] = -1 / 2
    b[count_row] = 0
    count_row += 1

    # if i == 0 and j == Ny:
    # -3/2 U0,Ny +2 U1,Ny -1/2 U2,Ny = 0
    A[count_row, idx(0, Ny)] = -3 / 2
    A[count_row, idx(1, Ny)] = 2
    A[count_row, idx(2, Ny)] = -1 / 2
    b[count_row] = 0
    count_row += 1

    # if i == Nx and j == 0:
    # 1/2 UNx-2,0 -2 UNx-1,0 +3/2 UNx,0 = 0
    A[count_row, idx(Nx - 2, 0)] = 1 / 2
    A[count_row, idx(Nx - 1, 0)] = -2
    A[count_row, idx(Nx, 0)] = 3 / 2
    b[count_row] = 0
    count_row += 1

    # if i == Nx and j == Ny:
    # 1/2 UNx-2,Ny -2 UNx-1,Ny +3/2 UNx,Ny = 0
    A[count_row, idx(Nx - 2, Ny)] = 1 / 2
    A[count_row, idx(Nx - 1, Ny)] = -2
    A[count_row, idx(Nx, Ny)] = 3 / 2
    b[count_row] = 0
    count_row += 1

    for i in range(2, Nx - 1):
        A[count_row, idx(i - 1, 0)] = 2
        A[count_row, idx(i, 0)] = -8
        A[count_row, idx(i + 1, 0)] = 2
        A[count_row, idx(i - 2, 1)] = 1
        A[count_row, idx(i - 1, 1)] = -8
        A[count_row, idx(i, 1)] = 21
        A[count_row, idx(i + 1, 1)] = -8
        A[count_row, idx(i + 2, 1)] = 1
        A[count_row, idx(i - 1, 2)] = 2
        A[count_row, idx(i, 2)] = -8
        A[count_row, idx(i + 1, 2)] = 2
        A[count_row, idx(i, 3)] = 1
        b[count_row] = 0
        count_row += 1

    for i in range(2, Nx - 1):
        A[count_row, idx(i, Ny - 3)] = 1
        A[count_row, idx(i - 1, Ny - 2)] = 2
        A[count_row, idx(i, Ny - 2)] = -8
        A[count_row, idx(i + 1, Ny - 2)] = 2
        A[count_row, idx(i - 2, Ny - 1)] = 1
        A[count_row, idx(i - 1, Ny - 1)] = -8
        A[count_row, idx(i, Ny - 1)] = 21
        A[count_row, idx(i + 1, Ny - 1)] = -8
        A[count_row, idx(i + 2, Ny - 1)] = 1
        A[count_row, idx(i - 1, Ny)] = 2
        A[count_row, idx(i, Ny)] = -8
        A[count_row, idx(i + 1, Ny)] = 2
        b[count_row] = 0
        count_row += 1

    for j in range(2, Ny - 1):
        A[count_row, idx(1, j - 2)] = 1
        A[count_row, idx(0, j - 1)] = 2
        A[count_row, idx(1, j - 1)] = -8
        A[count_row, idx(2, j - 1)] = 2
        A[count_row, idx(0, j)] = -8
        A[count_row, idx(1, j)] = 21
        A[count_row, idx(2, j)] = -8
        A[count_row, idx(3, j)] = 1
        A[count_row, idx(0, j + 1)] = 2
        A[count_row, idx(1, j + 1)] = -8
        A[count_row, idx(2, j + 1)] = 2
        A[count_row, idx(1, j + 2)] = 1
        b[count_row] = 0
        count_row += 1

    for j in range(2, Ny - 1):
        A[count_row, idx(Nx - 1, j - 2)] = 1
        A[count_row, idx(Nx - 2, j - 1)] = 2
        A[count_row, idx(Nx - 1, j - 1)] = -8
        A[count_row, idx(Nx, j - 1)] = 2
        A[count_row, idx(Nx - 3, j)] = 1
        A[count_row, idx(Nx - 2, j)] = -8
        A[count_row, idx(Nx - 1, j)] = 21
        A[count_row, idx(Nx, j)] = -8
        A[count_row, idx(Nx - 2, j + 1)] = 2
        A[count_row, idx(Nx - 1, j + 1)] = -8
        A[count_row, idx(Nx, j + 1)] = 2
        A[count_row, idx(Nx - 1, j + 2)] = 1
        b[count_row] = 0
        count_row += 1

    A[count_row, idx(0, 0)] = 2
    A[count_row, idx(1, 0)] = -8
    A[count_row, idx(2, 0)] = 2
    A[count_row, idx(0, 1)] = -8
    A[count_row, idx(1, 1)] = 22
    A[count_row, idx(2, 1)] = -8
    A[count_row, idx(3, 1)] = 1
    A[count_row, idx(0, 2)] = 2
    A[count_row, idx(1, 2)] = -8
    A[count_row, idx(2, 2)] = 2
    A[count_row, idx(1, 3)] = 1
    b[count_row] = 0
    count_row += 1

    A[count_row, idx(1, Ny - 3)] = 1
    A[count_row, idx(0, Ny - 2)] = 2
    A[count_row, idx(1, Ny - 2)] = -8
    A[count_row, idx(2, Ny - 2)] = 2
    A[count_row, idx(0, Ny - 1)] = -8
    A[count_row, idx(1, Ny - 1)] = 22
    A[count_row, idx(2, Ny - 1)] = -8
    A[count_row, idx(3, Ny - 1)] = 1
    A[count_row, idx(0, Ny)] = 2
    A[count_row, idx(1, Ny)] = -8
    A[count_row, idx(2, Ny)] = 2
    b[count_row] = 0
    count_row += 1

    A[count_row, idx(Nx - 2, 0)] = 2
    A[count_row, idx(Nx - 1, 0)] = -8
    A[count_row, idx(Nx, 0)] = 2
    A[count_row, idx(Nx - 3, 1)] = 1
    A[count_row, idx(Nx - 2, 1)] = -8
    A[count_row, idx(Nx - 1, 1)] = 22
    A[count_row, idx(Nx, 1)] = -8
    A[count_row, idx(Nx - 2, 2)] = 2
    A[count_row, idx(Nx - 1, 2)] = -8
    A[count_row, idx(Nx, 2)] = 2
    A[count_row, idx(Nx - 1, 3)] = 1
    b[count_row] = 0
    count_row += 1

    A[count_row, idx(Nx - 1, Ny - 3)] = 1
    A[count_row, idx(Nx - 2, Ny - 2)] = 2
    A[count_row, idx(Nx - 1, Ny - 2)] = -8
    A[count_row, idx(Nx, Ny - 2)] = 2
    A[count_row, idx(Nx - 3, Ny - 1)] = 1
    A[count_row, idx(Nx - 2, Ny - 1)] = -8
    A[count_row, idx(Nx - 1, Ny - 1)] = 22
    A[count_row, idx(Nx, Ny - 1)] = -8
    A[count_row, idx(Nx - 2, Ny)] = 2
    A[count_row, idx(Nx - 1, Ny)] = -8
    A[count_row, idx(Nx, Ny)] = 2
    b[count_row] = 0
    count_row += 1

    # Solve the linear system
    U, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Convert the solution to a 2D grid
    phi = U.reshape((Nx + 1, Ny + 1))

    # Level plot with 50 levels
    X, Y = np.meshgrid(x, y)

    plt.contourf(Y, X, phi, levels=50, cmap='viridis')  # Increase to 50 levels
    plt.colorbar()
    plt.title('Solution ($\Phi$) to the Biharmonic Equation')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.show()

    # Save the plot
    plt.savefig(f'biharmonic_{Nx}.png')

    # Contour plot with white contour lines
    plt.contourf(Y, X, phi, cmap='viridis')  # Filled contour plot
    plt.colorbar()  # Color bar for reference

    # Add contour lines over the filled contour plot
    plt.contour(Y, X, phi, colors='w', levels=50, linewidths=0.7)  # Example: red color lines with width of 2

    plt.title('Solution ($\Phi$) to the Biharmonic Equation')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.show()

    # Save the plot
    plt.savefig(f'biharmonic_contour_{Nx}.png')

    # Calculate gradients
    grad_y, grad_x = np.gradient(phi)
    u_x = grad_y
    u_y = -grad_x

    # Calculate the velocity magnitude
    velocity_magnitude = np.sqrt(u_x ** 2 + u_y ** 2)

    # Scale u_x and u_y for better visualization
    u_x_scale = u_x * 15
    u_y_scale = u_y * 15

    # Calculate the velocity magnitude
    velocity_magnitude_scale = np.sqrt(u_x_scale ** 2 + u_y_scale ** 2)

    # Create a meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Select a step size for slicing; higher numbers will have fewer vectors
    step = 10  # Adjust this to change the density of the vectors

    # Slice the arrays to reduce the number of vectors
    Y_slice = Y[::step, ::step]
    X_slice = X[::step, ::step]
    u_x_slice = u_x_scale[::step, ::step]
    u_y_slice = u_y_scale[::step, ::step]
    velocity_magnitude_slice = velocity_magnitude_scale[::step, ::step]

    # Plot the velocity field
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)  # Adjust figure size

    quiver = ax.quiver(Y_slice, X_slice, u_y_slice, u_x_slice, velocity_magnitude_slice,
                       angles='xy', scale_units='xy', scale=0.1, cmap=cm.jet)
    ax.set_aspect('equal')
    ax.set_title('Velocity Field')
    ax.set_xlabel('y')
    ax.set_ylabel('x')

    # Adding a colorbar to represent the velocity magnitude
    cbar = fig.colorbar(quiver, ax=ax)
    cbar.set_label('Velocity Magnitude')

    plt.show()

    # Save the plot
    plt.savefig(f'biharmonic_velocity_{Nx}.png')

    # Plot contours of the velocity magnitude
    fig, ax = plt.subplots()
    plt.contourf(Y, X, velocity_magnitude, cmap=cm.jet, levels=100)
    plt.colorbar()
    plt.title('Velocity Magnitude')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.show()

    # Save the plot
    plt.savefig(f'biharmonic_velocity_contour_{Nx}.png')




