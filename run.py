import tensorflow as tf
import matplotlib.pyplot as plt
import argparse


# hyperparameters
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WEIGHT = 224
EPOCHS = 20
NUM_CLASSES = 3
CHECKPOINT_PATH = "checkpoints/cp.ckpt"

# preprocess
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(IMG_HEIGHT, IMG_WEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='sparse')
val_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(IMG_HEIGHT, IMG_WEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='sparse')
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(IMG_HEIGHT, IMG_WEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='sparse')


# model
VGG16_MODEL = tf.keras.applications.VGG16()
VGG16_MODEL.trainable = False

model = tf.keras.Sequential([

    VGG16_MODEL,
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])


def train(model, train_data, val_data):
    """ Training routine. """
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                     monitor='val_accuracy',
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     verbose=0)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[cp_callback]
    )


def test(model, test_data):
    """ Testing routine. """
    model.load_weights(CHECKPOINT_PATH)
    # Run model on test set
    loss, acc = model.evaluate(
        x=test_data,
        verbose=1,
    )
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    class_names = ['circle', 'rectangle', 'triangle']
    plt.figure(figsize=(10, 10))
    images, labels = next(test_data)

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(class_names[int(labels[i])])
    plt.show()


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--test',
        action='store_true',
        help='''Skips training and evaluates on the test set once.''')

    return parser.parse_args()


# Make arguments global
ARGS = parse_args()


def main():
    """ Main function. """
    model.summary()
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    if ARGS.test:
        test(model, test_generator)

    else:
        train(model, train_generator, val_generator)


if __name__ == "__main__":
    main()
