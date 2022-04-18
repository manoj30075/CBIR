import numpy as np
import imageio.v2 as imageio
from skimage.feature import hog
from cache import Cache


def cosine_similarity(np_array_1, np_array_2):
    """
    Returns the cosine similarity of two numpy arrays
    """
    return np.dot(np_array_1, np_array_2) / (np.linalg.norm(np_array_1) * np.linalg.norm(np_array_2))


class HOG:
    def __init__(self, orient_size=8, cell_size=16, block_size=1, bin_size=10, slice_size=6):
        self.orient_size = orient_size
        self.cell_size = cell_size
        self.block_size = block_size
        self.bin_size = bin_size
        self.slice_size = slice_size
        self.all_features = None
        self.cache = Cache('hog')

    def _HOG(self, img, region='local', normalize=True):
        if region == 'global':
            pixel_per_cell = (self.cell_size, self.cell_size)
            cell_per_block = (self.block_size, self.block_size)
            fd, hog_image = hog(img, orientations=self.orient_size, pixels_per_cell=pixel_per_cell,
                                cells_per_block=cell_per_block, visualize=True, channel_axis=-1)
            bins = np.linspace(0, np.max(fd), self.bin_size + 1, endpoint=True)
            hist, _ = np.histogram(fd, bins=bins)
            if normalize:
                hist = np.array(hist) / np.sum(hist)
            return hist.flatten()
        else:
            height, width, _ = img.shape
            hist = np.zeros((self.slice_size, self.slice_size, self.bin_size))
            h_slice = np.around(np.linspace(0, height, self.slice_size + 1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, self.slice_size + 1, endpoint=True)).astype(int)
            for i in range(len(h_slice) - 1):
                for j in range(len(w_slice) - 1):
                    hog_image = img[h_slice[i]:h_slice[i + 1], w_slice[j]:w_slice[j + 1]]
                    hist[i][j] = self._HOG(hog_image, region='global', normalize=False)
            if normalize:
                hist = np.array(hist) / np.sum(hist)
            return hist.flatten()

    def _get_all_image_features(self):
        if self.all_features is None:
            self.all_features = self.cache.get_all_image_features()
        return self.all_features

    def _get_image_features(self, image_path):
        img = imageio.imread(image_path)
        return self._HOG(img)

    def save_image_features(self, image_path, image_name):
        fts = self._get_image_features(image_path)
        self.cache.cache(fts, image_name)

    def get_similar_images(self, image_path, n=10):
        fts = self._get_image_features(image_path)
        all_fts = self._get_all_image_features()
        sim_images = []
        for f in all_fts:
            sim_images.append((f['name'], cosine_similarity(fts, f['features'])))
        sim_images.sort(key=lambda x: x[1])
        return self._with_return_attributes(sim_images[:n])

    def _with_return_attributes(self, images):
        """
        Sets the image paths
        """
        result = []
        for image in images:
            row = {'url': image[0].replace('-', '/')[:-4], 'similarity': image[1], 'metric': 'cosine'}
            result.append(row)
        return result


def main():
    pass


if __name__ == '__main__':
    main()
