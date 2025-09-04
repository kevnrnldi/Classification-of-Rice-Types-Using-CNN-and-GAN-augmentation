# main.py
import os
import numpy as np
from data_preprocessing import preprocess_data
from train_gan import train_all_gans
from utils import save_synthetic_samples, make_archive

# Folder dataset dan kelas
data_dir = "Rice_Image_Dataset"
classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
img_size = 100
max_per_class = 1500

# Preprocessing data
X_train, X_test, y_train, y_test = preprocess_data(data_dir, classes, img_size, max_per_class)

# Melatih GAN untuk setiap kelas
generators = train_all_gans(X_train, classes, latent_dim=100, epochs=400, batch_size=32)

# Menghasilkan gambar sintetis untuk setiap kelas
num_synth = 500
synth_images = []
synth_labels = []

for label, generator in enumerate(generators):
    print(f"Generating synthetic images for class: {classes[label]}")
    noise = np.random.normal(0, 1, (num_synth, 100))
    imgs = generator.predict(noise)
    imgs = 0.5 * imgs + 0.5  # Convert from [-1,1] to [0,1]
    synth_images.append(imgs)
    synth_labels.append(np.full((num_synth,), label))

X_synth = np.concatenate(synth_images, axis=0)
y_synth_indices = np.concatenate(synth_labels, axis=0)

print(f"Generated {X_synth.shape[0]} synthetic images.")

# Save synthetic images
save_synthetic_samples(generators[0], epoch=0, n_samples=25)  # Save sample images for Arborio class

# Make an archive of synthetic data
make_archive("synthetic_data", "synthetic_dataset")
