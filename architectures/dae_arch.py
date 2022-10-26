from .ae_arch import AEArchitecture
from utils.training import save_checkpoint
from data_collection import DataCollection
import time
import random
import tensorflow as tf


class DAEArchitecture(AEArchitecture):

    def __init__(self, model, args):
        # model must be (Autoencoder, discriminator)
        super(DAEArchitecture, self).__init__(model, args)
        self.loss_func = None  # we want more spesific names
        self.optimizer = None  # we want more spesific names
        self.mse = tf.keras.losses.MeanSquaredError()
        self.ae_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-5)
        self.model = model[0]  # Auto encoder
        self.discriminator = model[1]  # discriminator

    def ae_loss(self, x, x_hat):
        return self.mse(x, x_hat)

    def discriminator_loss(self, real_output, fake_output, loss_weight):
        real_loss = self.mse(tf.ones_like(real_output),
                             real_output)  # so all latent variables strive to a value of 1 if real?
        fake_loss = self.mse(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return loss_weight * total_loss

    def generator_loss(self, fake_output, loss_weight):
        return loss_weight * tf.reduce_mean(self.mse(tf.ones_like(fake_output), fake_output))

    @tf.function
    def train_step(self, x, y):
        # Note that y is not used
        with tf.GradientTape() as ae_tape, \
                tf.GradientTape() as disc_tape, \
                tf.GradientTape() as gen_tape:
            x_hat = self.model(x)

            real_output, c0 = self.discriminator(x, training=True)
            fake_output, c1 = self.discriminator(x_hat, training=True)

            auto_loss = self.ae_loss(x, x_hat)
            disc_loss = self.discriminator_loss(real_output, fake_output, 1)
            gen_loss = self.generator_loss(fake_output, 1)

        gradients_of_ae = ae_tape.gradient(auto_loss, self.model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                        self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss,
                                                   self.model.decoder.trainable_variables)

        self.ae_optimizer.apply_gradients(zip(gradients_of_ae, self.model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.model.decoder.trainable_variables))
        return auto_loss, disc_loss, gen_loss

    def train(self, data_collection: DataCollection):
        if not data_collection.all_not_none(['normal_train_data']):
            raise ValueError('data_collection is missing normal_train_data')

        normal_train_data_dataset = tf.data.Dataset.from_tensor_slices(
            data_collection.normal_train_data).shuffle(self.buffer_size, seed=42).batch(self.batch_size)

        auto_epoch_losses, disc_epoch_losses, gen_epoch_losses = [], [], []
        train_start = time.time()
        for epoch in range(self.epochs):
            start = time.time()
            auto_epoch_loss, disc_epoch_loss, gen_epoch_loss = 0.0, 0.0, 0.0
            for image_batch in normal_train_data_dataset:
                auto_loss, disc_loss, gen_loss = self.train_step(image_batch, image_batch)
                auto_epoch_loss += auto_loss
                disc_epoch_loss += disc_loss
                gen_epoch_loss += gen_loss

            auto_epoch_loss /= len(normal_train_data_dataset)  # divide by number of batches
            disc_epoch_loss /= len(normal_train_data_dataset)  # divide by number of batches
            gen_epoch_loss /= len(normal_train_data_dataset)  # divide by number of batches
            auto_epoch_losses.append(auto_epoch_loss)
            disc_epoch_losses.append(disc_epoch_loss)
            gen_epoch_losses.append(gen_epoch_loss)
            with open(self.dir_path + '/auto_epoch_losses.txt', 'a+') as f:
                f.write(f'{auto_epoch_loss}\n')
            with open(self.dir_path + '/disc_epoch_losses.txt', 'a+') as f:
                f.write(f'{disc_epoch_loss}\n')
            with open(self.dir_path + '/gen_epoch_losses.txt', 'a+') as f:
                f.write(f'{gen_epoch_loss}\n')

            inds = random.sample(range(len(data_collection.normal_train_data)), self.num_samples)  # 10 random images
            data = data_collection.normal_train_data[inds]

            data_inferred = self.infer(data)
            self.save_data_images_ae(data, data_inferred, epoch)

            self.save_checkpoint(epoch, 'AE')
            self.save_checkpoint(epoch, 'DISC')
            self.print_epoch(epoch,
                             time.time() - start,
                             [auto_epoch_loss, disc_epoch_loss, gen_epoch_loss],
                             ['AE loss', 'DISC loss', 'GEN loss'])

        print(f'Total training time: {(time.time() - train_start) // 60} min')
        self.save_checkpoint(-1, 'AE')
        self.save_checkpoint(-1, 'DISC')
        self.save_training_metrics_image([auto_epoch_losses, disc_epoch_losses, gen_epoch_losses],
                                         ['AE loss', 'DISC loss', 'GEN loss'])

    def save_checkpoint(self, epoch, model_subtype=None):
        if model_subtype == 'AE':
            save_checkpoint(self.dir_path, self.model, model_subtype, epoch)
        if model_subtype == 'DISC':
            save_checkpoint(self.dir_path, self.discriminator, model_subtype, epoch)

    def load_checkpoint(self):
        path = f'{self.dir_path}/training_checkpoints/checkpoint_full_model_AE'
        self.model.load_weights(path)
        path = f'{self.dir_path}/training_checkpoints/checkpoint_full_model_DISC'
        self.discriminator.load_weights(path)
