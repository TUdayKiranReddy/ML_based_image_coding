# ICA based coding
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt



class ICA_compression():
    def __init__(self):
        self.n_components = 30
        self.compressor = FastICA(self.n_components)

    def learn(self, raw_datas):
        return self.compressor.fit(raw_datas)
        
    def compress(self, raw_data):
        return self.compressor.fit_transform(raw_data)

    def decompress(self, data):
        return self.compressor.inverse_transform(data)

    def compress_multiple(self, images):
        comp_images = []
        comp_images = [self.compress(images[i]) for i in range(len(images))]
        return np.array(comp_images)

    def decompress_multiple(self, comp_images):
        recon_images = []
        recon_images = [self.decompress(comp_images[i]) for i in range(len(comp_images))]
        return np.array(recon_images)
