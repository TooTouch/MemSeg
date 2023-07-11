"""
Various utility functions for:
    - parsing user arguments.
    - computing the area under a curve.
    - generating a toy dataset to test the evaluation script.
"""
from bisect import bisect
import numpy as np


def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x: Samples from the domain of the function to integrate
          Need to be sorted in ascending order. May contain the same value
          multiple times. In that case, the order of the corresponding
          y values will affect the integration with the trapezoidal rule.
        y: Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
          determined by interpolating between its neighbors. Must not lie
          outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("WARNING: Not all x and y values passed to trapezoid(...)"
              " are finite. Will continue with only the finite values.")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction

def generate_toy_dataset(num_images, image_width, image_height, gt_size):
    """Generate a toy dataset to test the evaluation script.

    Args:
        num_images: Number of images that the toy dataset contains.
        image_width: Width of the dataset images in pixels.
        image_height: Height of the dataset images in pixels.
        gt_size: Size of rectangular ground truth regions that are
          artificially generated on the dataset images.

    Returns:
        anomaly_maps: List of numpy arrays that contain random anomaly maps.
        ground_truth_map: Corresponding list of numpy arrays that specify a
          rectangular ground truth region.
    """
    # Fix a random seed for reproducibility.
    np.random.seed(1338)

    # Create synthetic evaluation data with random anomaly scores and
    # simple ground truth maps.
    anomaly_maps = []
    ground_truth_maps = []
    for _ in range(num_images):
        # Sample a random anomaly map.
        anomaly_map = np.random.random((image_height, image_width))

        # Construct a fixed ground truth map.
        ground_truth_map = np.zeros((image_height, image_width))
        ground_truth_map[0:gt_size, 0:gt_size] = 1

        anomaly_maps.append(anomaly_map)
        ground_truth_maps.append(ground_truth_map)

    return anomaly_maps, ground_truth_maps