import tensorflow as tf
import os
import numpy as np
from image_to_train import bands_to_image, display_image, unpack_numpy_subimages
import sys
import argparse
from model import get_model, get_loss, get_optimizer


'''
The paper says that the images were cropped to 41x41 images for training. Does this mean that our inputs
must be of size 41x41 and then recompiled???
'''


if __name__ == '__main__':
    EPOCHS = 80
    CLIP_NORM = 0.01
    BATCH_SIZE = 64

    X, Y = unpack_numpy_subimages('Train_subimages')
    print('subimage_shapes: {}, number of training subimages: {}'.format(X.shape, len(X)))
    print(bands_to_image(X[0]).shape)
    print(bands_to_image(X[0]))
    print(bands_to_image(X[0] + Y[0]))
    example_x = tf.expand_dims(bands_to_image(X[0]), axis=2)
    example_y = tf.expand_dims(bands_to_image(X[0]+Y[0]), axis=2)
    print('PSNR similarity:', tf.image.psnr(example_x, example_y, max_val=1.0).numpy())

    train_size = int(0.8*len(X))
    valid_size = int(0.15*len(X))

    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=BATCH_SIZE*10) # may need to increase since the same images are next to each other

    train_dataset = dataset.take(train_size)
    valid_dataset = dataset.skip(train_size).take(valid_size)
    test_dataset  = dataset.skip(train_size+valid_size)

    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset  = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    model = get_model()
    loss = get_loss()
    optimizer = get_optimizer()

    for epoch in range(EPOCHS):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(x_batch_train)
                loss_value = loss(y_batch_train, predictions)
            gradients = tape.gradient(loss_value, model.trainable_variables)

            # unsure if we should use norm or global norm
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, CLIP_NORM)
            optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

        # val_logits = model(valid_dataset)
        # val_loss = loss(y_valid, val_logits)
        # if epoch % 10 == 0:
        #     print(val_loss)

        # Validation loop
        total_val_loss = 0
        num_batches = 0
        for x_batch_val, y_batch_val in valid_dataset:
            # Predict
            val_predictions = model(x_batch_val, training=False)        # unsure about training=False

            # Calculate loss
            val_loss = loss(y_batch_val, val_predictions)

            # Accumulate loss and batch count
            total_val_loss += val_loss
            num_batches += 1

        # Calculate average validation loss
        avg_val_loss = total_val_loss / num_batches

        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss.numpy()}")






    # example_train_path = os.path.join('Train_subimages', '0901x2x.npy')
    # example_train = np.load(example_train_path)
    # print(example_train.shape)
    # im = bands_to_image(example_train[0]+example_train[1])
    # display_image(im)

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--train_directory', type=str, default='Train_bands'
    # )
    #
    # parser.add_argument(
    #     '--output_file', type=str, default='model_weights'
    # )
    # parser.add_argument(
    #     '--'
    # )
    #

    # model_output = 'weights'

