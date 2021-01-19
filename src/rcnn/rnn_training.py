import os
import itertools
import codecs
import editdistance
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers import Reshape, Lambda, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.optimizers import Adadelta
import keras
import random
import copy
import imgaug.augmenters as iaa
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
import cv2
import glob

np.random.seed(55)
OUTPUT_DIR = 'model'
alphabet = u"abcdđefghijklmnopqrstuvwxyzABCDĐEFGHIJKLMNOPQRSTUVWXYZ àảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬÈẺẼÉẸÊỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢÙỦŨÚỤƯỪỬỮỨỰỲỶỸÝỴ0123456789'-.:,"
fonts = glob.glob(r"fonts/*")
backs = []
for filename in glob.glob(r"back/*.png"):
    backs.append(cv2.imread(filename))


def add_real_data(img_name, w, h):
    path = 'real_data/'
    img = cv2.imread(path + img_name)
    new_h = random.randint(25, 32)
    new_w = int(1.0 * img.shape[1] * new_h / img.shape[0])
    max_shift_x = w - new_w
    max_shift_y = h - new_h
    while max_shift_x < 0 or max_shift_y < 0:
        new_h -= 1
        new_w = int(1.0 * img.shape[1] * new_h / img.shape[0])
        max_shift_x = w - new_w
        max_shift_y = h - new_h

    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    top_left_x = np.random.randint(0, int(max_shift_x) + 1)
    top_left_y = np.random.randint(0, int(max_shift_y) + 1)
    back = np.zeros((h, w, 3))
    back[top_left_y:top_left_y + new_h, top_left_x: top_left_x + new_w, :] = new_img
    return back


def paint_text(text, w, h, noise=10):
    size = random.randint(15, 30)
    font = ImageFont.truetype(np.random.choice(fonts), size)
    ascent, descent = font.getmetrics()
    (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
    height = ascent + descent
    max_shift_x = w - width
    max_shift_y = h - height
    while max_shift_x < 0 or max_shift_y < 0:
        size -= 1
        font = ImageFont.truetype(np.random.choice(fonts), size)
        ascent, descent = font.getmetrics()
        (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
        height = ascent + descent
        max_shift_x = w - width
        max_shift_y = h - height

    top_left_x = np.random.randint(0, int(max_shift_x) + 1)
    top_left_y = np.random.randint(0, int(max_shift_y) + 1)
    back = copy.copy(backs[random.randint(0, len(backs) - 1)][:, :w, :])
    img = Image.fromarray(back)
    draw = ImageDraw.Draw(img)
    color = np.ones(3) * random.randint(0, 100) + noise
    color -= np.random.randint(2 * noise, size=3)
    color[color < 0] = 0
    b, g, r = color.astype(int)
    draw.text((top_left_x, top_left_y), text, font=font, fill=(b, g, r, 0))
    back = np.array(img)
    return back


def augment_data(images):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    augumenters = iaa.Sequential([
        iaa.SomeOf(n=(0, 5), children=[
            iaa.Invert(0.2, 0.5),
            iaa.Add(value=(-40, 40), per_channel=0.3),
            iaa.Multiply(mul=(0.8, 1.2), per_channel=0.3),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.0, 0.5)),
                iaa.AverageBlur(k=(3, 15)),
                iaa.MedianBlur(k=(3, 15))
            ]),
            iaa.AdditiveGaussianNoise(scale=0.05 * 255, per_channel=0.3),
            iaa.Dropout(p=(0, 0.2), per_channel=0.3),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.ContrastNormalization((0.5, 1.5), per_channel=0.3),
            iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
            iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                       translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, rotate=(-3, 3), shear=(-5, 5))
        ], random_order=True),
        sometimes(iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=["constant", "edge"], pad_cval=(0, 128)))
    ], random_order=True)
    augumenters = augumenters.to_deterministic()  #
    newDatas = augumenters.augment_images(images)
    return newDatas


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports numpy.array and list objects.')
    return ret


# characters to integer
def text_to_labels(text):
    ret = []
    for char in text:
        label = alphabet.find(char)
        if label < 0:
            continue
        ret.append(label)

    return ret


# numerical to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet) or c == -1:
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def get_output_size():
    return len(alphabet) + 1


class TextImageGenerator(keras.callbacks.Callback):
    def __init__(self, monogram_file, bigram_file, name_file, field_file, real_file, batch_size, img_w, img_h,
                 downsample_factor, val_split, absolute_max_string_len=40):
        self.batch_size = batch_size
        self.img_w = img_w
        self.img_h = img_h
        self.monogram_file = monogram_file
        self.bigram_file = bigram_file
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.blank_label = get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len
        self.name_file = name_file
        self.field_file = field_file
        self.real_file = real_file
        self.real_text = None

    def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.2, name_fraction=0.4):
        assert max_string_len <= self.absolute_max_string_len
        assert num_words % self.batch_size == 0
        assert (self.val_split * num_words) % self.batch_size == 0

        self.num_words = num_words
        self.string_list = [''] * self.num_words
        tmp_string_list = []
        self.max_string_len = max_string_len

        if (self.real_text is None):
            with codecs.open(self.real_file, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                self.real_img_path = []
                self.real_text = []
                self.real_data = np.ones([len(lines), self.absolute_max_string_len]) * -1
                self.real_len = [0] * len(lines)
                i = 0
                for line in lines:
                    line = line.replace("\n", "")
                    path, text = line.split(r"|")
                    # print(i, path, text)
                    self.real_img_path.append(path)
                    self.real_text.append(text)
                    self.real_len[i] = len(text)
                    self.real_data[i, :len(text)] = text_to_labels(text)
                    i += 1

        with codecs.open(self.monogram_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines:
                if len(tmp_string_list) == int(self.num_words * mono_fraction):
                    break
                word = ""
                for ch in line:
                    if alphabet.find(ch) >= 0:
                        word += ch

                word = word.split()
                if len(word) >= 1 and (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(" ".join(word))

        with codecs.open(self.name_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines:
                if len(tmp_string_list) == int(self.num_words * (mono_fraction + name_fraction)):
                    break
                word = ""
                for ch in line:
                    if alphabet.find(ch) >= 0:
                        word += ch
                word = word.split()
                # word =
                if len(word) >= 1 and (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(" ".join(word))

        with codecs.open(self.bigram_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines:
                if len(tmp_string_list) == self.num_words:
                    break
                word = ""
                for ch in line:
                    if alphabet.find(ch) >= 0:
                        word += ch

                word = word.split()
                if len(word) > 1 and (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(" ".join(word))

        if len(tmp_string_list) < self.num_words:
            raise IOError('Could not pull enough words from supplied monogram and bigram files.')

        with codecs.open(self.field_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            self.field_list = []
            self.field_len = []
            self.field_data = np.ones([len(lines) + 1, self.absolute_max_string_len]) * -1
            random.shuffle(lines)
            i = 0
            for line in lines:
                word = ""
                for ch in line:
                    if alphabet.find(ch) >= 0:
                        word += ch
                if len(word) >= 1:
                    self.field_list.append(word)
                    self.field_len.append(len(word))
                    self.field_data[i, :len(word)] = text_to_labels(word)
                    i += 1
            self.field_list.append("")
            self.field_len.append(1)
            self.field_data[i, 0] = self.blank_label

        random.shuffle(tmp_string_list)
        self.string_list = tmp_string_list

        X_text = []
        Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        Y_len = [0] * self.num_words
        for i, word in enumerate(self.string_list):
            Y_len[i] = len(word)
            Y_data[i, :len(word)] = text_to_labels(word)
            X_text.append(word)

        self.X_text = X_text
        self.Y_data = Y_data
        self.Y_len = np.expand_dims(np.array(Y_len), 1)
        self.cur_val_index = self.val_split
        self.cur_train_index = 0

        del X_text
        del Y_data
        del Y_len

    def get_batch(self, index, batch_size, train):
        X_data = np.zeros([batch_size, self.img_h, self.img_w, 3])
        labels = np.ones([batch_size, self.absolute_max_string_len])
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        source_str = []
        images = []
        for i in range(batch_size - 1):
            if train and i > batch_size - 4:
                rand_index = random.randint(0, len(self.field_list))
                if rand_index == len(self.field_list):
                    string = str(random.randint(100000000, 999999999))
                    img = self.paint_func(string)
                    X_data[i] = img
                    images.append(img)
                    string_label = np.ones(self.absolute_max_string_len) * -1
                    string_label[:len(string)] = text_to_labels(string)
                    labels[i, :] = string_label
                    input_length[i] = self.img_w / self.downsample_factor - 1
                    label_length[i] = len(string)
                    source_str.append(string)
                else:
                    img = self.paint_func(self.field_list[rand_index])
                    X_data[i] = img
                    images.append(img)
                    labels[i, :] = self.field_data[rand_index]
                    input_length[i] = self.img_w / self.downsample_factor - 1
                    label_length[i] = self.field_len[rand_index]
                    source_str.append(self.field_list[rand_index])
            else:
                img = self.paint_func(self.X_text[index + i])
                images.append(img)
                X_data[i] = img
                labels[i, :] = self.Y_data[(index + i) % len(self.Y_data)]
                input_length[i] = self.img_w / self.downsample_factor - 1
                label_length[i] = self.Y_len[(index + i) % len(self.Y_len)]
                source_str.append(self.X_text[(index + i) % len(self.X_text)])

        real_index = random.randint(0, len(self.real_data) - 1)
        img = add_real_data(self.real_img_path[real_index], self.img_w, self.img_h)
        img = img.astype(np.uint8)
        images.append(img)
        labels[batch_size - 1, :] = self.real_data[real_index]
        label_length[batch_size - 1] = self.real_len[real_index]
        input_length[batch_size - 1] = self.img_w / self.downsample_factor - 1
        source_str.append(self.real_text[real_index])

        images = augment_data(images)
        X_data = np.asarray(images)

        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str,
                  'images': images
                  }
        outputs = {'ctc': np.zeros([batch_size])}  # dummy data for dummy loss function
        return inputs, outputs

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.batch_size, train=True)
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = 0
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.batch_size, train=False)
            self.cur_val_index += self.batch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split
            yield ret

    def on_train_begin(self, logs=None):
        self.build_word_list(24000, -1, 0.2, 0.4)
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h)

    def on_epoch_begin(self, epoch, logs=None):
        K.set_learning_phase(True)
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h)
        if epoch % 25 == 20:
            self.build_word_list(24000, -1, 0.2, 0.4)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_decoder(args):
    y_pred, input_length = args
    seq_len = tf.squeeze(input_length, axis=1)
    return K.ctc_decode(y_pred=y_pred, input_length=seq_len, greedy=True, beam_width=100, top_paths=1)[0]


def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


class VizCallback(keras.callbacks.Callback):
    def __init__(self, run_name, test_func, text_img_gen, num_display_words=12, decode_func=None):
        self.test_func = test_func
        self.decode_func = decode_func
        self.output_dir = OUTPUT_DIR + "/" + run_name
        # print (self.output_dir, 'test')
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            # print("num_proc", num_proc)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f' % (
            num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs=None):
        K.set_learning_phase(False)
        if epoch % 5 == 0:
            self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % epoch))
            self.show_edit_distance(1024)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        predicted = self.test_func([word_batch['the_input'][:self.num_display_words]])[0]
        # print(predicted.shape)
        ctc_res = self.decode_func([predicted, word_batch["input_length"][:self.num_display_words]])[0]
        for i in range(self.num_display_words):
            print("Truth = '%s' \tDecoded = '%s'\tCtc_decoder = '%s'" % (
                word_batch['source_str'][i], res[i], labels_to_text(ctc_res[i])))


def train(run_name, start_epoch, stop_epoch):
    # Input Parameters
    img_w = 512
    img_h = 32
    words_per_epoch = 6400
    val_split = 0.2
    val_words = int(words_per_epoch * val_split)
    batch_size = 32
    pool_size = 2
    img_gen = TextImageGenerator(monogram_file='word_mono.txt',
                                 bigram_file="words.txt",
                                 field_file="field.txt",
                                 name_file="name.txt",
                                 real_file="real_data.txt",
                                 batch_size=batch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=(pool_size ** 3),
                                 val_split=words_per_epoch - val_words
                                 )

    input_shape = (img_h, img_w, 3)
    # Make Network
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    # Convolution layer
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
    inner = Dropout(0.2, name="dropout1")(inner, training=True)

    inner = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                   name='conv5')(inner)
    inner = BatchNormalization()(inner, training=True)
    inner = Dropout(0.2, name="dropout2")(inner, training=True)
    inner = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                   name='conv6')(inner)
    inner = BatchNormalization()(inner, training=True)
    inner = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='max4')(inner)

    inner = Conv2D(512, (2, 2), strides=(1, 1), activation="relu", kernel_initializer='he_normal', name='conv7')(inner)

    # Model(inputs=input_data, outputs=inner).summary()

    # CNN to RNN
    conv_to_rnn_dims = (63, 512)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # RNN layer
    lstm1 = LSTM(256, return_sequences=True, kernel_initializer="he_normal", name='lstm1')(inner)
    lstm1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer="he_normal", name='lstm1b')(inner)
    merge_concat = concatenate([lstm1, lstm1b])

    # transforms RNN output to character activations
    inner = Dense(get_output_size(), kernel_initializer="he_normal", name="Dense")(merge_concat)
    y_pred = Activation("softmax", name="softmax")(inner)

    # Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.summary()

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    ada = Adadelta(decay=1e-3)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        model.load_weights(weight_file, True)

    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])
    decoder = Lambda(ctc_decoder, output_shape=[None, ], name="decoder")([y_pred, input_length])
    decode_func = K.function([y_pred, input_length], [decoder])
    viz_cb = VizCallback(run_name, test_func, img_gen.next_val(), decode_func=decode_func)

    model.fit_generator(generator=img_gen.next_train(),
                        steps_per_epoch=(words_per_epoch - val_words) // batch_size,
                        epochs=stop_epoch,
                        validation_data=img_gen.next_val(),
                        validation_steps=val_words // batch_size,
                        callbacks=[viz_cb, img_gen],
                        initial_epoch=start_epoch)


if __name__ == '__main__':
    run_name = "1"
    train(run_name, 0, 10000)
