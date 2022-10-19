import librosa
import tensorflow as tf
import numpy as np
from conf import *

SAVED_MODEL_PATH = "myModel.h5"
cnf = conf()
conf1 = cnf.get_default_conf()


class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.

    :param model: Trained model
    """

    model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    # model = tf.keras.models.load_model('model')
    _mapping = {
        0: 'akawuka',
        1: 'banana',
        2: 'obulwadde',
        3: 'nnyaanya',
        4: 'pampu',
        5: 'obutunda',
        6: 'plantation',
        7: 'ensujju',
        8: 'okulimibwa',
        9: 'mpeke',
        10: 'okusaasaana',
        11: 'ebigimusa',
        12: 'ekikolo',
        13: 'farm',
        14: 'kisaanyi',
        15: 'kikajjo',
        16: 'ekisaanyi',
        17: 'ndwadde',
        18: 'omusiri',
        19: 'butterfly',
        20: 'munyeera',
        21: 'eggobe',
        22: 'ebiwojjolo',
        23: 'ebisoolisooli',
        24: 'namuginga',
        25: 'okugimusa',
        26: 'maize streak virus',
        27: 'ekirime',
        28: 'miceere',
        29: 'sikungula',
        30: 'lumonde',
        31: 'okukungula',
        32: 'cassava',
        33: 'ebirime',
        34: 'ebijanjaalo',
        35: 'weeding',
        36: 'garden',
        37: 'drought',
        38: 'leaves',
        39: 'insect',
        40: 'akatungulu',
        41: 'seed',
        42: 'pepper',
        43: 'matooke seedlings',
        44: 'harvesting',
        45: 'medicine',
        46: 'nursery bed',
        47: 'mucungwa',
        48: 'endwadde',
        49: 'pawpaw',
        50: 'enkota',
        51: 'ensiringanyi',
        52: 'kassooli',
        53: 'okufuuyira',
        54: 'caterpillars',
        55: 'ekijanjaalo',
        56: 'okukkoola',
        57: 'crop',
        58: 'okulima',
        59: 'endagala',
        60: 'kaamulali',
        61: 'ennima',
        62: 'omuceere',
        63: 'micungwa',
        64: 'ebisaanyi',
        65: 'plant',
        66: 'eddagala',
        67: 'ennimiro',
        68: 'amakoola',
        69: 'ebiwuka',
        70: 'ekigimusa',
        71: 'bibala',
        72: 'beans',
        73: 'nnimiro',
        74: 'ebinyebwa',
        75: 'passion fruit',
        76: 'Spinach',
        77: 'okuzifuuyira',
        78: 'ekirwadde',
        79: 'nakavundira',
        80: 'nfukirira',
        81: 'onion',
        82: 'ddagala',
        83: 'muwogo',
        84: 'irrigate',
        85: 'akasaanyi',
        86: 'ekikajjo',
        87: 'emmwanyi',
        88: 'ekiwojjolo',
        89: 'orange',
        90: 'ebibala',
        91: 'ebyobulimi',
        92: 'ensuku',
        93: 'farmer',
        94: 'spray',
        95: 'obumonde',
        96: 'nnasale beedi',
        97: 'abalimi',
        98: 'okusaasaanya',
        99: 'doodo',
        100: 'enva endiirwa',
        101: 'ebikolo',
        102: 'obusaanyi',
        103: 'omulimisa',
        104: 'muceere',
        105: 'ejjobyo',
        106: 'ebikajjo',
        107: 'omucungwa',
        108: 'amappapaali',
        109: 'ensigo',
        110: 'ebikoola',
        111: 'emboga',
        112: 'spread',
        113: 'akamonde',
        114: 'kasaanyi',
        115: 'dig',
        116: 'ebisooli',
        117: 'nnakati',
        118: 'obulimi',
        119: 'mangoes',
        120: 'sweet potatoes',
        121: 'akammwanyi',
        122: 'vegetables',
        123: 'worm',
        124: 'amakungula',
        125: 'omuyembe',
        126: 'harvest',
        127: 'olusuku',
        128: 'amalagala',
        129: 'npk',
        130: 'kikolo',
        131: 'maize',
        132: 'coffee',
        133: 'ebijjanjalo',
        134: 'irish potatoes',
        135: 'ebimera',
        136: 'matooke',
        137: 'leaf',
        138: 'afukirira',
        139: 'ensukusa',
        140: 'caterpillar',
        141: 'sukumawiki',
        142: 'suckers',
        143: 'amatooke',
        144: 'emiyembe',
        145: 'endokwa',
        146: 'okusimba',
        147: 'mulimi',
        148: 'farming instructor',
        149: 'fertilizer',
        150: 'kukungula',
        151: 'akatunda',
        152: 'omulimi',
        153: 'nambaale',
        154: 'ebikongoliro',
        155: 'sow',
        156: 'ground nuts',
        157: 'super grow',
        158: 'ekimera',
        159: 'fruit picking',
        160: 'obuwuka',
        161: 'okusiga',
        162: 'emisiri',
        163: 'ekitooke',
        164: 'emicungwa',
        165: 'pumpkin',
        166: 'greens',
        167: 'bulimi',
        168: 'agriculture',
        169: 'okufukirira',
        170: 'tomatoes',
        171: 'fruit',
        172: 'ebitooke',
        173: 'rice',
        174: 'ebbugga',
        175: 'ppaapaali',
        176: 'okunnoga',
        177: 'obutungulu',
        178: 'ennyaanya',
        179: 'lusuku',
        180: 'insects',
        181: 'mango',
        182: 'eppapaali',
        183: 'Pump',
        184: 'maize stalk borer',
        185: 'ekibala',
        186: 'watermelon',
        187: 'ekyeya',
        188: 'disease',
        189: 'ekikoola',
        190: 'faamu',
        191: 'cabbages',
        192: 'sugarcane'}
    _instance = None

    def predict(self, file_path):
        """

        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract spectogram
        X = self.preprocess_audio(file_path)
        print(X.shape)
        # reshape spec

        data = X.reshape(1, 128, 221, 1)
        print(type(data))

        print(data.shape)

        # convert to tensor
        newData = tf.convert_to_tensor(data, dtype="float32")
        print(newData.shape)

        # convert X to np array
        newDataArray = np.array(newData)
        print(newDataArray.shape)

        # get the predicted label
        # print(self.model.summary())
        predictions = self.model.predict(newDataArray)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword

    def melspectogram_dB(self, file_path, cst=3, top_db=80.):
        row_sound, sr = librosa.load(file_path, sr=conf1.sampling_rate)
        sound = np.zeros((cst*sr,))
        if row_sound.shape[0] < cst*sr:
            sound[:row_sound.shape[0]] = row_sound[:]
        else:
            sound[:] = row_sound[:cst*sr]

        spec = librosa.feature.melspectrogram(sound,
                                              sr=conf1.sampling_rate,
                                              n_mels=conf1.n_mels,
                                              hop_length=conf1.hop_length,
                                              n_fft=conf1.n_fft,
                                              fmin=conf1.fmin,
                                              fmax=conf1.fmax)
        spec_db = librosa.power_to_db(spec)
        spec_db = spec_db.astype(np.float32)

        return spec_db

    def spec_to_image(self, spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_img = 255 * (spec_norm - spec_min) / (spec_max - spec_min)

        return spec_img.astype(np.uint8)

    def preprocess_audio(self, audio_path):
        spec = self.melspectogram_dB(audio_path)
        spec = self.spec_to_image(spec)
        return spec


def Keyword_Spotting_Service(self):
    """Factory function for Keyword_Spotting_Service class.

    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
    _Keyword_Spotting_Service.model = tf.keras.models.load_model(
        SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    # import os
    # print("Dir log: ", os.getcwd())
    # cur_dir = os.getcwd()

    # keyword = kss.predict(os.path.join(cur_dir, "test", "obutunda.wav"))
    # print(keyword)
