from keras import backend as K
from keras.models import Model
import numpy as np
import cv2
import glob
import tensorflow as tf
from keras.layers import Input, Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM

alphabet = u"abcdđefghijklmnopqrstuvwxyzABCDĐEFGHIJKLMNOPQRSTUVWXYZ àảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬÈẺẼÉẸÊỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢÙỦŨÚỤƯỪỬỮỨỰỲỶỸÝỴ0123456789'-.:,"


def labels_to_text(labels):
    ret = []

    for c in labels:
        if c == len(alphabet) or c == -1:  # CTC blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def ctc_decoder(args):
    y_pred, input_length = args
    seq_len = tf.squeeze(input_length, axis=1)
    return K.ctc_decode(y_pred=y_pred, input_length=seq_len, greedy=True, beam_width=100, top_paths=2)[0]


class Recognize(object):
    def __init__(self):
        input_data = Input(name='the_input', shape=(32, None, 3), dtype='float32')
        inner = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max1')(inner)
        inner = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max2')(inner)
        inner = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                       name='conv3')(inner)
        inner = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                       name='conv4')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max3')(inner)
        inner = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                       name='conv5')(inner)
        inner = BatchNormalization()(inner, training=False)
        inner = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                       name='conv6')(inner)
        inner = BatchNormalization()(inner, training=False)
        inner = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='max4')(inner)
        CNN_out = Conv2D(512, (2, 2), strides=(1, 1), activation="relu", kernel_initializer='he_normal', name='conv7')(
            inner)

        input_rnn = Input(shape=(None, 512), dtype='float32')
        lstm1 = LSTM(256, return_sequences=True, kernel_initializer="he_normal", name='lstm1')(input_rnn)
        lstm1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer="he_normal", name='lstm1b')(
            input_rnn)
        merge_concat = concatenate([lstm1, lstm1b])

        inner = Dense(len(alphabet) + 1, kernel_initializer="he_normal", name="Dense")(merge_concat)
        y_pred = Activation("softmax", name="softmax")(inner)

        path_to_weight_model = "../model/weights.h5"

        self.CNN = Model(input_data, CNN_out)
        self.CNN.load_weights(path_to_weight_model, True)

        self.RNN = Model(input_rnn, y_pred)
        self.RNN.load_weights(path_to_weight_model, True)

        input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.decoder = Lambda(ctc_decoder, output_shape=[None, ], name="decoder")([y_pred, input_length])
        self.decode = K.function([y_pred, input_length], [self.decoder])

    def read_word(self, word_image):
        if word_image.shape[0] > word_image.shape[1] or word_image.shape[0] == 0 or word_image.shape[1] == 0:
            return ""

        new_width = int(word_image.shape[1] * 32.0 / word_image.shape[0])
        img_gau = word_image
        img = cv2.resize(img_gau, (new_width, 32), interpolation=cv2.INTER_CUBIC)
        if new_width < 32:
            return ""

        out_cnn = self.CNN.predict(np.asarray([img]))
        input_length = np.ones([1, 1]) * (new_width // 8 - 1)
        out_cnn = out_cnn.reshape((out_cnn.shape[0], new_width // 8 - 1, 512))
        predicted = self.RNN.predict(out_cnn)

        out = self.decode([predicted, input_length])[0][0]

        return labels_to_text(out)


if __name__ == "__main__":
    recog = Recognize()
    for file in glob.glob("test_word/*.png"):
        print(file)
        img = cv2.imread(file)
        print(recog.read_word(img))
