import numpy as np


class ToolSet:
    def __init__(self, rle_string):
        self.rle_string = rle_string

    def rle_to_mask(self, width, height):
        """
        convert RLE(run length encoding) string to numpy array

        Parameters:
        rle_string (str): string of rle encoded mask
        height (int): height of the mask
        width (int): width of the mask

        Returns:
        numpy.array: numpy array of the mask
        """
        rows, cols = height, width

        if self.rle_string == -1:
            return np.zeros((height, width))
        else:
            rle_numbers = [int(num_string) for num_string in self.rle_string.split(' ')]
            rle_pairs = np.array(rle_numbers).reshape(-1, 2)
            img = np.zeros(rows * cols, dtype=np.uint8)
            for index, length in rle_pairs:
                index -= 1
                img[index:index + length] = 255
            img = img.reshape(cols, rows)
            img = img.T
            return img
