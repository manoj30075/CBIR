import imageio.v2 as imageio
import numpy as np
from cache import Cache

n_bin = 256


class RGB:
    __slots__ = ['cache', 'all_features']

    def __init__(self):
        self.cache = Cache('rgb')
        self.all_features = None

    def histogram(self, image):
        result = []
        for channel_id in range(3):
            hist, _ = np.histogram(
                image[:, :, channel_id], bins=256, range=(0, 256), normed=True
            )
            result.append(hist)

        return result

    def get_similar_images(self, file_path, n=10):
        image = imageio.imread(file_path)
        all_features = self._get_all_image_features()
        query_hist = self.histogram(image)
        results = []
        for idx, sample in enumerate(all_features):
            s_name, s_features = sample['name'], sample['features']
            similarity = []
            for i in range(3):
                similarity.append(1-cosine_similarity(s_features[i], query_hist[i]))
            distance = np.mean(similarity)
            results.append({
                'url': s_name[:-4],
                'similarity': distance,
                'metric': 'cosine'
            })
        results.sort(key=lambda x: x['similarity'])
        if n < len(results):
            results = results[:n]
        return results

    def _get_all_image_features(self):
        if self.all_features is None:
            self.all_features = self.cache.get_all_image_features()
        return self.all_features

    def get_image_features(self, file_path):
        image = imageio.imread(file_path)
        return self.histogram(image)

    def save_image_features(self, file_path, image_name):
        feature = self.get_image_features(file_path)
        self.cache.cache(feature, image_name)


def cosine_similarity(np_array_1, np_array_2):
    """
    Returns the cosine similarity of two numpy arrays
    """
    return np.dot(np_array_1, np_array_2) / (np.linalg.norm(np_array_1) * np.linalg.norm(np_array_2))
