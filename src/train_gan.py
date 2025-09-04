# src/train_gan.py
import numpy as np
from gan_model import build_generator, build_discriminator, build_gan
from tensorflow.keras.models import load_model
import os

def train_gan_for_class(X_train_class, epochs=400, batch_size=32, latent_dim=100):
    # Build model
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
        metrics=['accuracy']
    )
    gan = build_gan(generator, discriminator)

    for epoch in range(epochs):
        half_batch = batch_size // 2

        # Training Discriminator on real and fake images
        idx = np.random.randint(0, X_train_class.shape[0], half_batch)
        real_imgs = X_train_class[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_imgs = generator.predict(noise, verbose=0)

        # Train discriminator on real images
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        # Train discriminator on fake images
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Training Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))  # Label for the generator: 'valid' (1)
        g_loss = gan.train_on_batch(noise, valid_y)

        # Print the progress
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs} | D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}% | G loss: {g_loss:.4f}")
    
    return generator

def train_all_gans(X_train, classes, latent_dim=100, epochs=400, batch_size=32):
    generators = []
    for class_index, class_name in enumerate(classes):
        X_train_class = X_train[np.argmax(y_train, axis=1) == class_index]
        print(f"Training GAN for class: {class_name}")
        generator = train_gan_for_class(X_train_class, epochs, batch_size, latent_dim)
        generators.append(generator)
    return generators
