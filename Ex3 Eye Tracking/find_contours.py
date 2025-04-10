import numpy as np

def find_contours(binary_image):
    """
    Implements a basic version of `cv2.findContours()` to detect contours in a binary image.
    
    Args:
        binary_image (numpy array): A binary (black & white) image with objects as white (255) and background as black (0).
    
    Returns:
        contours (list of numpy arrays): List of detected contours, where each contour is an array of (x, y) points.
    """
    height, width = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)  # To track visited pixels
    contours = []

    # Define the 8 possible directions (clockwise starting from right)
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), 
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    def is_valid(x, y):
        """Check if the coordinate is within the image boundaries."""
        return 0 <= x < height and 0 <= y < width

    def trace_contour(start_x, start_y):
        """Traces a single contour using Moore-Neighbor Tracing algorithm."""
        contour = []
        x, y = start_x, start_y

        # Start direction is right (0 index in directions list)
        start_dir = 0
        prev_x, prev_y = x, y

        while True:
            contour.append((y, x))  # Store as (col, row) for consistency

            # Try moving in the current direction and search clockwise
            for i in range(8):
                dir_index = (start_dir + i) % 8
                dx, dy = directions[dir_index]
                nx, ny = x + dx, y + dy

                if is_valid(nx, ny) and binary_image[nx, ny] == 255 and not visited[nx, ny]:
                    visited[nx, ny] = True
                    start_dir = (dir_index + 6) % 8  # Move two steps back to start search correctly
                    x, y = nx, ny
                    break
            else:
                # If no new point is found, we are at the end of the contour
                break

            # If we return to the starting point, stop tracing
            if (x, y) == (start_x, start_y) and (prev_x, prev_y) != (start_x, start_y):
                break

            prev_x, prev_y = x, y

        return np.array(contour)

    
    # Scan the image to find contours
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 255 and not visited[i, j]:  # Unvisited white pixel
                visited[i, j] = True
                contour = trace_contour(i, j)
                if len(contour) > 5:  # Filter out small/noise contours
                    contours.append(contour)

    return contours


import numpy as np
from scipy.spatial.distance import directed_hausdorff

def find_contours(binary_image, merge_threshold=20.0):
    """
    Implements a basic version of `cv2.findContours()` to detect contours in a binary image.
    
    Args:
        binary_image (numpy array): A binary (black & white) image with objects as white (255) and background as black (0).
        merge_threshold (float): Maximum Hausdorff distance to consider two contours as identical.
    
    Returns:
        contours (list of numpy arrays): List of detected contours, where each contour is an array of (x, y) points.
    """
    height, width = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)  # To track visited pixels
    contours = []

    # Define the 8 possible directions (clockwise starting from right)
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), 
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    def is_valid(x, y):
        """Check if the coordinate is within the image boundaries."""
        return 0 <= x < height and 0 <= y < width

    def trace_contour(start_x, start_y):
        """Traces a single contour using Moore-Neighbor Tracing algorithm."""
        contour = []
        x, y = start_x, start_y

        # Start direction is right (0 index in directions list)
        start_dir = 0
        prev_x, prev_y = x, y

        while True:
            contour.append((y, x))  # Store as (col, row) for consistency

            # Try moving in the current direction and search clockwise
            for i in range(8):
                dir_index = (start_dir + i) % 8
                dx, dy = directions[dir_index]
                nx, ny = x + dx, y + dy

                if is_valid(nx, ny) and binary_image[nx, ny] == 255 and not visited[nx, ny]:
                    visited[nx, ny] = True
                    start_dir = (dir_index + 6) % 8  # Move two steps back to start search correctly
                    x, y = nx, ny
                    break
            else:
                # If no new point is found, we are at the end of the contour
                break

            # If we return to the starting point, stop tracing
            if (x, y) == (start_x, start_y) and (prev_x, prev_y) != (start_x, start_y):
                break

            prev_x, prev_y = x, y

        return np.array(contour)

    def is_duplicate(new_contour, existing_contours, threshold):
        """Checks if a new contour is a near duplicate of any existing contour."""
        for existing in existing_contours:
            if directed_hausdorff(new_contour, existing)[0] < threshold and directed_hausdorff(existing, new_contour)[0] < threshold:
                return True
        return False

    # Scan the image to find contours
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 255 and not visited[i, j]:  # Unvisited white pixel
                visited[i, j] = True
                contour = trace_contour(i, j)
                if len(contour) > 5:  # Filter out small/noise contours
                    if not is_duplicate(contour, contours, merge_threshold):
                        contours.append(contour)

    return contours
