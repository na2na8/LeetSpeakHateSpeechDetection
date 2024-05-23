import os
import numpy as np
from PIL import Image
from gensim.models import KeyedVectors

arial_path = '/home/nykim/HateSpeech/02_images/Arial-Unicode-MS'
pingfang_path = '/home/nykim/HateSpeech/02_images/PingFang-SC-Regular'

arial_broken = np.array(Image.open(os.path.join('/home/nykim/HateSpeech/03_code/VIPER/02_arial_broken.ppm')).convert("L"))
pingfang_broken = np.array(Image.open(os.path.join('/home/nykim/HateSpeech/03_code/VIPER/02_pingfang_broken.ppm')).convert("L"))

arial_ppms = os.listdir(arial_path)
pingfang_ppms = os.listdir(pingfang_path)

def get_vectors(path, broken, ppms) :
    words = []
    vectors = []
    for ppm in ppms :
        img = Image.open(os.path.join(path, ppm)).convert("L")
        img = np.array(img).flatten()
        if np.array_equal(broken, img) : continue
        unicode_idx = chr(int(ppm[:-4]))
        words.append(unicode_idx)
        vectors.append(img/255.)
    return words, vectors

if __name__ == "__main__" :
    words = []
    vectors = []
    
    # arial
    arial_w, arial_v = get_vectors(arial_path, arial_broken, arial_ppms)
    # pingfang
    pingfang_w, pingfang_v = get_vectors(pingfang_path, pingfang_broken, pingfang_ppms)
    
    words += arial_w
    words += pingfang_w
    vectors += arial_v
    vectors += pingfang_v
    
    model = KeyedVectors(24*24)
    model.add_vectors(words, vectors)
    
    model.save('/home/nykim/HateSpeech/03_code/VIPER/02_model.bin')