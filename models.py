import hyperparameters as hp
import tensorflow as tf
from tensorflow import keras
import tensorflow_graphics.nn.loss.chamfer_distance as tfg_chamfer
from tensorflow.keras.layers import \
    Conv2D, Activation, BatchNormalization, Activation, Layer
tf.config.run_functions_eagerly(True)


class MaxPool2DWithArgmax(Layer):
    def __init__(self, ksize=2, **kwargs):
        super(MaxPool2DWithArgmax, self).__init__(**kwargs)
        self.ksize = ksize

    def call(self, inputs):
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs, ksize=[1,self.ksize,self.ksize,1], strides=[1,self.ksize,self.ksize,1], padding='VALID')

        return output, argmax


class MaxUpPool2DWithIndices(Layer):
    def __init__(self, ksize=2, **kwargs):
        super(MaxUpPool2DWithIndices, self).__init__(**kwargs)
        self.ksize = ksize

    def call(self, inputs):

        input, indices = inputs[0], inputs[1]
        input_size = tf.size(input)
        input_shape = tf.shape(input, out_type="int32")

        output_shape = (input_shape[0], input_shape[1]*self.ksize,
                        input_shape[2]*self.ksize, input.shape[3])

        indices = tf.cast(indices, tf.int32)

        ones = tf.ones_like(indices, dtype="int32")
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(
            tf.range(output_shape[0], dtype="int32"), shape=batch_shape)
        channel_range = tf.range(output_shape[3], dtype="int32")

        b = ones * batch_range
        y = indices // (output_shape[2]*output_shape[3])
        x = indices % (output_shape[2]*output_shape[3]) // output_shape[3]
        c = ones * channel_range

        indices_values = tf.reshape(tf.stack([b, y, x, c]), (4, input_size))
        indices_values = tf.transpose(indices_values)
        input_values = tf.reshape(input, [input_size])
        
        output = tf.scatter_nd(indices_values, input_values, output_shape)
        
        return output


class EncoderLayer(Layer):
    def __init__(self, filters=64, lay_num=2, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.filters = filters
        self.lay_num = lay_num

        self.conv1 = Conv2D(filters, 3, 1, padding="same")
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')

        self.conv2 = Conv2D(filters, 3, 1, padding="same")
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')

        if lay_num == 3:
            self.conv3 = Conv2D(filters, 3, 1, padding="same")
            self.bn3 = BatchNormalization()
            self.act3 = Activation('relu')

        self.mp = MaxPool2DWithArgmax(2)

    def call(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.act1(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.act2(outputs)

        if self.lay_num == 3:
            outputs = self.conv3(outputs)
            outputs = self.bn3(outputs)
            outputs = self.act3(outputs)

        outputs, indices = self.mp(outputs)

        return outputs, indices


class DecoderLayer(Layer):
    def __init__(self, in_filters=64, out_filters=64, lay_num=2, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.lay_num = lay_num

        self.mup = MaxUpPool2DWithIndices(2)
        self.conv1 = Conv2D(in_filters, 3, 1, padding="same")
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')

        if lay_num == 2:
            self.conv2 = Conv2D(out_filters, 3, 1, padding="same")
            self.bn2 = BatchNormalization()
            self.act2 = Activation('relu')

        elif lay_num == 3:
            self.conv2 = Conv2D(in_filters, 3, 1, padding="same")
            self.bn2 = BatchNormalization()
            self.act2 = Activation('relu')

            self.conv3 = Conv2D(out_filters, 3, 1, padding="same")
            self.bn3 = BatchNormalization()
            self.act3 = Activation('relu')

    def call(self, inputs):

        outputs = self.mup(inputs)

        outputs = self.conv1(outputs)
        outputs = self.bn1(outputs)
        outputs = self.act1(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.act2(outputs)

        if self.lay_num == 3:
            outputs = self.conv3(outputs)
            outputs = self.bn3(outputs)
            outputs = self.act3(outputs)

        return outputs


class Encoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.encoder1 = EncoderLayer(64, 2, name='encoder_1')
        self.encoder2 = EncoderLayer(128, 2, name='encoder_2')
        self.encoder3 = EncoderLayer(256, 3, name='encoder_3')
        self.encoder4 = EncoderLayer(512, 3, name='encoder_4')
        self.encoder5 = EncoderLayer(512, 3, name='encoder_5')

    def call(self, inputs):

        outputs, indices1 = self.encoder1(inputs)
        outputs, indices2 = self.encoder2(outputs)
        outputs, indices3 = self.encoder3(outputs)
        outputs, indices4 = self.encoder4(outputs)
        outputs, indices5 = self.encoder5(outputs)

        return outputs, indices1, indices2, indices3, indices4, indices5


class Decoder(tf.keras.Model):
    def __init__(self, out_channels, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.decoder1 = DecoderLayer(512, 512, 3, name='decoder_1')
        self.decoder2 = DecoderLayer(512, 256, 3, name='decoder_2')
        self.decoder3 = DecoderLayer(256, 128, 3, name='decoder_3')
        self.decoder4 = DecoderLayer(128, 64, 2, name='decoder_4')
        self.decoder5 = DecoderLayer(64, out_channels, 2, name='decoder_5')

    def call(self, inputs):

        outputs = self.decoder1([inputs[0], inputs[1]])
        outputs = self.decoder2([outputs, inputs[2]])
        outputs = self.decoder3([outputs, inputs[3]])
        outputs = self.decoder4([outputs, inputs[4]])
        outputs = self.decoder5([outputs, inputs[5]])

        return outputs


class SegNet(tf.keras.Model):
    def __init__(self, out_channels, load_vgg):
        super(SegNet, self).__init__()

        self.out_channels = out_channels
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.encoder = Encoder()
        self.decoder = Decoder(out_channels)

        if load_vgg:
            self.encoder.trainable = False

    def call(self, inputs):

        outputs, indices1, indices2, indices3, indices4, indices5 = self.encoder(
            inputs)

        outputs = self.decoder(
            [outputs, indices5, indices4, indices3, indices2, indices1])

        return outputs

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """
        loss = 0
        batch_size = min(hp.BATCH_SIZE, labels.shape[0])
        for i in range(batch_size):
            loss += tf.norm((labels[i, :, :, 0:3] -
                             predictions[i, :, :, 0:3]), axis=-1)
            loss += tf.norm((labels[i, :, :, 3:6] -
                             predictions[i, :, :, 3:6]), axis=-1)
            loss += tf.norm((labels[i, :, :, 6:9] -
                             predictions[i, :, :, 6:9]), axis=-1)
        return tf.reduce_mean(loss)/3.0/batch_size


class ChamferDistance(keras.metrics.Metric):
    def __init__(self, name="chamfer_distance", **kwargs):
        super(ChamferDistance, self).__init__(name=name, **kwargs)
        self.chamfer = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        points_num = 8000
        reshape = (y_true.shape[0], y_true.shape[1]*y_true.shape[2], y_true.shape[3])

        y_true = tf.reshape(y_true, reshape)
        y_pred = tf.reshape(y_pred, reshape)

        y_true_eval = tf.concat([y_true[:, :, 0:3], y_true[:, :, 3:6]], axis=1)
        y_pred_eval = tf.concat([y_pred[:, :, 0:3], y_pred[:, :, 3:6]], axis=1)
        
        choices_true = tf.random.shuffle(tf.range(y_true_eval.shape[1]))[:points_num]
        choices_pred = tf.random.shuffle(tf.range(y_pred_eval.shape[1]))[:points_num]

        y_true_eval = tf.gather(y_true_eval, choices_true, axis=1)
        y_pred_eval = tf.gather(y_pred_eval, choices_pred, axis=1)
        
        self.chamfer = tf.reduce_mean(tfg_chamfer.evaluate(y_true_eval, y_pred_eval))*100

    def result(self):
        
        return self.chamfer

    def reset_states(self):
        self.chamfer = 0
