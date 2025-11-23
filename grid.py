import numpy as np
import re
from pathlib import Path
from scipy.ndimage import maximum_filter

class Grid:
    """
    Represents a grid of tiles with north/east bounds and corresponding
    per-tile min/max values.
    """

    def __init__(self, north_min, north_max, east_min, east_max,
                 min_vals=None, max_vals=None, tile_size_km=1, tile_size_px=2500):
        self.north_min = north_min
        self.north_max = north_max
        self.east_min = east_min
        self.east_max = east_max

        self.tile_size_km = tile_size_km

        n_rows = north_max - north_min + 1
        n_cols = east_max - east_min + 1

        self.min_vals = (
            np.full((n_rows, n_cols), np.nan, dtype=np.float32)
            if min_vals is None else min_vals
        )
        self.max_vals = (
            np.full((n_rows, n_cols), np.nan, dtype=np.float32)
            if max_vals is None else max_vals
        )

    @classmethod
    def from_info_files(cls, folder_path, north_min, north_max, east_min, east_max, tile_size_km=1, tile_size_px=2500):
        """
        Creates a Grid by parsing .info files within the given bounds.

        Returns:
            Grid instance or None if no files found.
        """
        folder = Path(folder_path)
        n_rows = north_max - north_min + 1
        n_cols = east_max - east_min + 1

        min_vals = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        max_vals = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

        re_min = re.compile(r"Min(?:imum)?\s*=\s*([-+]?[0-9]*\.?[0-9]+)")
        re_max = re.compile(r"Max(?:imum)?\s*=\s*([-+]?[0-9]*\.?[0-9]+)")

        tiles_found = False

        for north in range(north_min, north_max + 1):
            for east in range(east_min, east_max + 1):
                filename = f"DTM_{tile_size_km}km_{north}_{east}.info"
                path = folder / filename
                if not path.exists():
                    continue

                text = path.read_text(errors="ignore")

                m_min = re_min.search(text)
                m_max = re_max.search(text)
                if not (m_min and m_max):
                    continue

                min_val = float(m_min.group(1))
                max_val = float(m_max.group(1))
                tiles_found = True

                col = east - east_min
                row = north_max - north  # north decreases downward

                min_vals[row, col] = min_val
                max_vals[row, col] = max_val

        if not tiles_found:
            # print("⚠️ No .info tiles found in the given bounds.")
            max_vals[:] = 10000;
            min_vals[:] = 0;

        return cls(north_min, north_max, east_min, east_max, min_vals, max_vals)

    def shape(self):
        """Returns (n_rows, n_cols) of the grid."""
        return self.min_vals.shape

    def missing_mask(self):
        """Returns a boolean mask where either min or max is NaN."""
        return np.isnan(self.min_vals) | np.isnan(self.max_vals)

    def __repr__(self):
        return (f"Grid(north={self.north_min}-{self.north_max}, "
                f"east={self.east_min}-{self.east_max}, "
                f"shape={self.shape()}, "
                f"valid_tiles={np.sum(~self.missing_mask())})")

    def max_with_neighbors(self, array_name="max_vals", filter_size=3):
        """
        Returns a new 2D NumPy array where each element is the maximum value
        within its 3×3 neighborhood (including itself).

        Parameters:
            array_name (str): Which data array to process ("max_vals" or "min_vals")

        Returns:
            np.ndarray: 2D array of the same shape with neighborhood maxima.
        """
        if array_name not in ("max_vals", "min_vals"):
            raise ValueError("array_name must be 'max_vals' or 'min_vals'")

        data = getattr(self, array_name)
        if data is None:
            raise ValueError(f"{array_name} is not defined in this Grid")

        # Handle NaNs by temporarily replacing with very low values
        nan_mask = np.isnan(data)
        filled = np.where(nan_mask, -np.inf, data)

        # Apply 3×3 max filter
        result = maximum_filter(filled, size=filter_size, mode="nearest")

        # Restore NaNs where all surrounding values were NaN
        all_nan_mask = maximum_filter(~nan_mask, size=3, mode="nearest") == 0
        result[all_nan_mask] = np.nan

        return result

    def get_highline_mask_by_max_min(self, H):
        mask = np.greater(self.max_with_neighbors() - self.min_vals, H)
        return mask

    def get_highline_coords(self, H):
        """
        Returns a list of (north, east) coordinates where the highline mask is True.
        """
        mask = self.get_highline_mask_by_max_min(H)
        rows, cols = np.where(mask)

        # Convert row/col indices back to geographic coordinates
        norths = self.north_max - rows
        easts = self.east_min + cols

        return list(zip(norths, easts))


