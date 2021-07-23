from unetgan.train_big import train_ubiggan
from unetgan.test_model import test_model
from unetgan.test_single import test_single

if __name__ == "__main__":
    # Only for 100k iters
    train_ubiggan(train_path=".", latent_dim=140, num_epochs=80)
    # test_model()
    # test_single()
