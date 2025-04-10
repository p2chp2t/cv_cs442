import numpy as np

def recover_ellipse_parameters(A, B, C, D, E, F):
    """Recover ellipse center, axes, and rotation angle from implicit equation."""
    # See https://math.stackexchange.com/questions/280937/finding-the-angle-of-rotation-of-an-ellipse-from-its-general-equation-and-the-ot
    
    if B == 0:
        theta = 0  # No rotation needed
    else:
        theta = 0.5 * np.arctan2(B, A - C)  # Rotation angle
    
    cos_t = np.sqrt((1 + np.cos(2 * theta)) / 2)
    sin_t = np.sqrt((1 - np.cos(2 * theta)) / 2) * np.sign(theta)
    
    # Compute rotated coefficients
    A_prime = A * cos_t**2 + B * cos_t * sin_t + C * sin_t**2
    C_prime = A * sin_t**2 - B * cos_t * sin_t + C * cos_t**2
    D_prime = D * cos_t + E * sin_t
    E_prime = -D * sin_t + E * cos_t
    F_prime = F

    # Compute ellipse center in rotated coordinates
    x0_prime = -D_prime / (2 * A_prime)
    y0_prime = -E_prime / (2 * C_prime)

    # Compute semi-axes lengths
    term = 4 * A_prime * C_prime - D_prime**2
    a2 = (C_prime * D_prime**2 + A_prime * E_prime**2 - 4 * A_prime * C_prime * F_prime) / (4 * A_prime**2 * C_prime)
    b2 = (C_prime * D_prime**2 + A_prime * E_prime**2 - 4 * A_prime * C_prime * F_prime) / (4 * A_prime * C_prime**2)
    
    if a2 <= 0 or b2 <= 0:
        return None, None, None  # Invalid ellipse

    a = np.sqrt(a2)
    b = np.sqrt(b2)

    # Rotate center back
    x0 = x0_prime * cos_t - y0_prime * sin_t
    y0 = x0_prime * sin_t + y0_prime * cos_t

    return (x0, y0), (a, b), theta

def plot_ellipse(A, B, C, D, E, F, ax, color='r', linestyle='-'):
    """Plot the fitted full ellipse using parametric equations."""
    ellipse_params = recover_ellipse_parameters(A, B, C, D, E, F)
    if ellipse_params is None:
        print("Invalid ellipse parameters")
        return
    
    center, axes, angle = ellipse_params

    # Generate full ellipse points
    t = np.linspace(0, 2 * np.pi, 400)  # Full ellipse
    ellipse_x = axes[0] * np.cos(t)
    ellipse_y = axes[1] * np.sin(t)

    # Rotate and translate
    R = np.array([[np.cos(angle), -np.sin(angle)], 
                  [np.sin(angle),  np.cos(angle)]])  # Rotation matrix
    ellipse_pts = R @ np.vstack([ellipse_x, ellipse_y]) + np.array(center).reshape(2, 1)

    # Plot full ellipse
    ax.plot(ellipse_pts[0, :], ellipse_pts[1, :], color=color, linestyle=linestyle, linewidth=4.0)