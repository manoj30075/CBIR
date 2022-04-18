# Import the libraries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np

from cache import Cache


def get_similarity(features, target_feature):
    """
    Get the similarity of the target feature to all the features in the list.
    """
    # Calculate the cosine similarity
    return np.linalg.norm(features - target_feature, axis=1)


def _with_return_attributes(images):
    """
    Sets the image paths
    """
    result = []
    for image in images:
        row = {'url': image.replace('-', '/')[:-4], 'similarity': 0, 'metric': 'cosine'}
        result.append(row)
    return result


class Vgg16:
    def __init__(self):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        self.cache = Cache('vgg16')
        self.all_features = None

    def extract(self, img):
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)

    def _get_all_image_features(self):
        if self.all_features is None:
            self.all_features = self.cache.get_all_image_features()
        return self.all_features

    def get_image_features(self, file_path):
        img = Image.open(file_path)
        feature = self.extract(img)
        return feature

    def save_image_features(self, file_path, image_name):
        feature = self.get_image_features(file_path)
        self.cache.cache(feature, image_name)

    def get_similar_images(self, file_path, n=5):
        feature = self.get_image_features(file_path)
        features_info = self._get_all_image_features()
        features = [feature_info['features'] for feature_info in features_info]
        dists = get_similarity(np.array(features), feature)
        sorted_indices = np.argsort(dists)[:n]
        sorted_features_info = [features_info[i]['name'] for i in sorted_indices]
        return _with_return_attributes(sorted_features_info)

