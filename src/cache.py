import numpy as np
import os

folder_path_dict = {
    'hog': '/Users/manojreddy/My Repos/Github/CBIR/features/hog',
    'rgb': '/Users/manojreddy/My Repos/Github/CBIR/features/rgb',
    'vgg16': '/Users/manojreddy/My Repos/Github/CBIR/features/vgg16'
}

folder_path_dict_ubuntu = {
    'hog': '/home/ubuntu/CBIR/features/hog',
    'rgb': '/home/ubuntu/CBIR/features/rgb',
    'vgg16': '/home/ubuntu/CBIR/features/vgg16'
}


def create_dir(folder_path):
    """
    Creates a directory
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


class Cache:
    """
    Class that saves and loads data
    """

    def __init__(self, feature_type):
        self.folder_path = folder_path_dict_ubuntu[feature_type]
        create_dir(self.folder_path)

    def cache(self, features, image_name):
        """
        Saves data to a file
        """
        image_name = image_name.replace('/', '-')
        np.save(os.path.join(self.folder_path, image_name + '.npy'), features)

    def get_all_image_features(self):
        """
        Loads all image features
        """
        features = []
        for image_name in os.listdir(self.folder_path):
            features.append({'name': image_name, 'features': np.load(os.path.join(self.folder_path, image_name))})
        return features


