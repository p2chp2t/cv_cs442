import numpy as np
import numpy.typing as npt
import scipy


def normalize_0_1(data: npt.NDArray) -> npt.NDArray:
    data = data - data.min()
    data = data / data.max()

    return data


def create_image():
    img = np.zeros([128, 128])
    img[30:-30, 20:30] = 1
    img[90:110, 40:-40] = 1

    np.save("simple_pattern", img)


def grid_sample(
    array: npt.NDArray, grid: npt.NDArray, default_value: float = 0
) -> npt.NDArray:
    """
    Perform grid-based sampling and interpolation on input tensor.
    Modified from: https://stackoverflow.com/a/79498402

    Args:
        array (np.ndarray): Input array with shape (H,W) of the input image.
        grid (np.ndarray): Sampling grid with shape (H,W,2).
        default_value (float, optional): Fill value for out-of-bound coordinates. Defaults to 0.

    Returns:
        np.ndarray: Transformed array with shape (H,W).
    """

    array = array[None, None]
    grid = grid[None]

    b, c = array.shape[:2]
    input_image_shape = np.array(array.shape[2:])
    b_ = grid.shape[0]
    output_image_shape = np.array(grid.shape[1:-1])
    grid_vec_dim = grid.shape[-1]
    assert b == b_ and len(input_image_shape) == grid_vec_dim
    out = []
    for t, g in zip(array, grid):
        out.append(
            np.stack(
                [
                    scipy.ndimage.map_coordinates(
                        input=t[i],
                        coordinates=(
                            (g.reshape([-1, grid_vec_dim]) + 1)
                            / 2
                            * (input_image_shape[::-1] - 1)
                        ).T[::-1],
                        order=1,
                        mode="constant",
                        cval=default_value,
                    )
                    for i in range(c)
                ]
            )
        )
    return np.concatenate(out, axis=0).reshape(b, c, *output_image_shape)[0, 0]
