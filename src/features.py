import os
import urllib.request as urllib
from hog import HOG
from rgb import RGB
from vgg16 import Vgg16
from cache import Cache
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_DATA_DIR = '/Users/manojreddy/My Repos/Github/CBIR/database'
URL = 'https://animals-cbir.b-cdn.net/database/'


class Features:

    def __init__(self):
        self.hog = HOG()
        self.rgb = RGB()
        self.vgg16 = Vgg16()

    def get_model(self, feature_type):
        if feature_type == 'hog':
            return self.hog
        elif feature_type == 'rgb':
            return self.rgb
        elif feature_type == 'vgg16':
            return self.vgg16

    def generate_and_save_features(self, feature_type, folder_path=DEFAULT_DATA_DIR):
        """
        Generates features for the data
        """
        # get list of files
        model = self.get_model(feature_type)
        print(model)
        sub_folders = os.listdir(folder_path)
        for sub_folder in sub_folders:
            # get list of files in sub_folder
            files = os.listdir(os.path.join(folder_path, sub_folder))
            for file in files:
                full_path = os.path.join(folder_path, sub_folder, file)
                print('Generating features for {}'.format(full_path))
                model.save_image_features(full_path, os.path.join(sub_folder, file))

    def get_similar_images_web(self, image_path, feature_type, n=10):
        model = self.get_model(feature_type)
        self.download_and_save_image(image_path)
        results = model.get_similar_images('query.jpg', n)
        for result in results:
            result['url'] = URL + result['url'].replace('-', '/')
        return results

    def download_and_save_image(self, url):
        urllib.urlretrieve(url, 'query.jpg')


if __name__ == '__main__':
    features = Features()
    features.generate_and_save_features('vgg16')
    # path = '/Users/manojreddy/My Repos/Github/CBIR/database/antelope/0a37838e99.jpg'
    # features.vgg16.get_similar_images(path, 10)