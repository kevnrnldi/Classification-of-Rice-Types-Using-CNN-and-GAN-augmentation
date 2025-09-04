import matplotlib.pyplot as plt
import shutil
import os

def save_synthetic_samples(generator, epoch, n_samples=25, latent_dim=100, output_dir='synthetic_data'):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    gen_images = generator.predict(noise)
    gen_images = (gen_images + 1) / 2.0
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_samples):
        plt.imshow(gen_images[i, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.savefig(f"{output_dir}/synthetic_{epoch}_{i}.png")
        plt.close()

def make_archive(folder="synthetic_data", output="synthetic_dataset"):
    shutil.make_archive(output, 'zip', folder)
    print(f"Arsip berhasil dibuat: {output}.zip")
