import tensorflow as tf
import os
import matplotlib.pyplot as plt
import argparse
from preprocess import Datasets
from models import SegNet, ChamferDistance
import hyperparameters as hp
import wget


def train(model, train_data, val_data):
    """ Training routine. """

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=hp.CHECKPOINT_PATH,
                                                     monitor='val_loss',
                                                     save_best_only=False,
                                                     save_weights_only=True,
                                                     verbose=0)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=hp.EPOCHS,
        callbacks=[cp_callback]
    )


def test(model, test_data, checkpoint):
    """ Testing routine. """

    model.load_weights(checkpoint)
    # Run model on test set
    loss, chamfer = model.evaluate(
        x=test_data,
        verbose=1,
    )
    

    images, labels = iter(test_data).get_next()[
        0], iter(test_data).get_next()[1]

    predictions = model.predict(images)

    for i in range(hp.BATCH_SIZE):

        plt.subplot(hp.BATCH_SIZE, 7, 1+7*i)
        plt.axis('off')
        plt.imshow(images[i])
        plt.title('Input Views')

        plt.subplot(hp.BATCH_SIZE, 7, 2+7*i)
        plt.axis('off')
        plt.imshow(predictions[i, :, :, 0:3])
        plt.title('NOCS (Prediction)')

        plt.subplot(hp.BATCH_SIZE, 7, 3+7*i)
        plt.axis('off')
        plt.imshow(labels[i, :, :, 0:3])
        plt.title('NOCS (Gound Truth)')

        plt.subplot(hp.BATCH_SIZE, 7, 4+7*i)
        plt.axis('off')
        plt.imshow(predictions[i, :, :, 3:6])
        plt.title('X-NOCS (Prediction)')

        plt.subplot(hp.BATCH_SIZE, 7, 5+7*i)
        plt.axis('off')
        plt.imshow(labels[i, :, :, 3:6])
        plt.title('X-NOCS (Gound Truth)')

        plt.subplot(hp.BATCH_SIZE, 7, 6+7*i)
        plt.axis('off')
        plt.imshow(predictions[i, :, :, 6:9])
        plt.title('Depth Peeling (Prediction)')

        plt.subplot(hp.BATCH_SIZE, 7, 7+7*i)
        plt.axis('off')
        plt.imshow(labels[i, :, :, 6:9])
        plt.title('Depth Peeling (Gound Truth)')

    plt.show()


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--test',
        default=None,
        help='''Skips training and evaluates on the test set once.''')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        help='''Path to pre-trained VGG-16 file.''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to start checkpoint.''')
    return parser.parse_args()


# Make arguments global
ARGS = parse_args()


def main():
    """ Main function. """
    datasets = Datasets()

    if ARGS.load_vgg:
        if not os.path.exists(ARGS.load_vgg):
            wget.download(
                "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)
        model = SegNet(out_channels=9, load_vgg=True)
        model(tf.keras.Input(shape=(480, 640, 3)))

        model.encoder.load_weights(ARGS.load_vgg, by_name=True)
    else:
        model = SegNet(out_channels=9, load_vgg=False)
        model(tf.keras.Input(shape=(480, 640, 3)))

        
    if ARGS.load_checkpoint:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)
        model.load_weights(ARGS.load_checkpoint, by_name=False)

    model.summary()
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=[ChamferDistance()])
    if ARGS.test:

        ARGS.test = os.path.abspath(ARGS.test)
        test(model, datasets.test_data, ARGS.test)

    else:
        train(model, datasets.train_data, datasets.val_data)


if __name__ == "__main__":
    main()
