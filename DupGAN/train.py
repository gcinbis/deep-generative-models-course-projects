import logging

import torch.optim as optim
from torchvision.utils import make_grid

from models import *
import os

from utils import generic_dataloader, SubsetWithIndices


class EncoderClassifierTrainer:

    def __init__(self, device, params, svhn_trainsplit, svhn_testsplit, mnist_trainsplit, mnist_testsplit,
                 summary_writer=None, experiment_name='encoder_classifier', ckpt_file=None, ckpt_root=os.path.join('ckpts', '')):

        self.device = device

        self.confidence_threshold = params.encoder_classifier_confidence_threshold
        self.ckpt_root = ckpt_root
        self.experiment_name = experiment_name
        self.summary_writer = summary_writer

        self.svhn_trainsplit_loader = generic_dataloader(self.device, svhn_trainsplit, shuffle=True,
                                                         batch_size=params.batch_size)
        self.svhn_testsplit_loader = generic_dataloader(self.device, svhn_testsplit, shuffle=False,
                                                        batch_size=params.batch_size)
        self.mnist_trainsplit_loader = generic_dataloader(self.device, mnist_trainsplit, shuffle=False,
                                                          batch_size=params.batch_size)
        self.mnist_testsplit_loader = generic_dataloader(self.device, mnist_testsplit, shuffle=False,
                                                         batch_size=params.batch_size)

        self.epoch = -1
        self.encoder_classifier = EncoderClassifier(self.device, in_channels=3).to(self.device)

        # Adam params taken from unit paper (except lr which is doubled for fast learning
        self.optimizer = optim.Adam(self.encoder_classifier.parameters(), lr=params.encoder_classifier_adam_lr,
                                    betas=(params.encoder_classifier_adam_beta1,
                                           params.encoder_classifier_adam_beta2))
        self.criterion = nn.CrossEntropyLoss()

        if ckpt_file is not None:
            self.load(ckpt_file)

        self.encoder_classifier.train()

    def load(self, ckpt_file):
        # adapted from https://pytorch.org/tutorials/beginner/saving_loading_models.html
        checkpoint = torch.load(ckpt_file, map_location=self.device)
        self.encoder_classifier.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.experiment_name = checkpoint['experiment_name']

    def save(self):
        torch.save({
            'model': 'encoder_classifier',
            'experiment_name': self.experiment_name,
            'epoch': self.epoch,
            'model_state_dict': self.encoder_classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            os.path.join(self.ckpt_root, '%s_%d.tar' % (self.experiment_name, self.epoch)))

    def evaluate(self, data_loader):

        # Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        self.encoder_classifier.eval()
        with torch.no_grad():
            loss = 0
            correct = 0
            total = 0
            above_confidence = 0
            for data in data_loader:

                inputs, labels, *_ = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                classifier_out, _ = self.encoder_classifier(inputs)
                loss += self.criterion(classifier_out, labels).item()
                confidence, predicted = torch.max(nn.functional.softmax(classifier_out, 1), 1)
                confidence = confidence.squeeze()
                above_confidence += len(confidence[confidence >= self.confidence_threshold])
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100.0*correct/total
            above_confidence_accuracy = 100.0*above_confidence/total
        self.encoder_classifier.train()

        return loss, accuracy, above_confidence, above_confidence_accuracy

    def train_until_epoch(self, last_epoch, eval_every_n_epochs=1, save_every_n_epochs=1):

        self.encoder_classifier.train()

        for self.epoch in range(self.epoch+1, last_epoch):

            logging.info('epoch %d',self.epoch)

            for data in self.svhn_trainsplit_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                classifier_out, _ = self.encoder_classifier(inputs)
                loss = self.criterion(classifier_out, labels)
                loss.backward()
                self.optimizer.step()

            # logging and saving stuff

            if self.epoch % eval_every_n_epochs == eval_every_n_epochs-1:

                svhn_train_loss, svhn_train_accuracy, *_ = self.evaluate(self.svhn_trainsplit_loader)
                svhn_test_loss, svhn_test_accuracy, *_ = self.evaluate(self.svhn_testsplit_loader)

                mnist_train_loss, mnist_train_accuracy, mnist_train_above_threshold,\
                mnist_train_above_threshold_accuracy = self.evaluate(self.mnist_trainsplit_loader)

                mnist_test_loss, mnist_test_accuracy, mnist_test_above_threshold,\
                mnist_test_above_threshold_accuracy = self.evaluate(self.mnist_testsplit_loader)

                self.summary_writer.add_scalars('loss', {'svhn_train_loss': svhn_train_loss,
                                                         'svhn_test_loss': svhn_test_loss,
                                                         'mnist_train_loss': mnist_train_loss,
                                                         'mnist_test_loss': mnist_test_loss}, self.epoch)

                self.summary_writer.add_scalars('accuracy_or_percentage',
                                                {'svhn_train_accuracy': svhn_train_accuracy,
                                                 'svhn_test_accuracy': svhn_test_accuracy,
                                                 'mnist_train_accuracy': mnist_train_accuracy,
                                                 'mnist_test_accuracy': mnist_test_accuracy,
                                                 'mnist_train_above_threshold': mnist_train_above_threshold_accuracy,
                                                 'mnist_test_above_threshold': mnist_test_above_threshold_accuracy},
                                                self.epoch)
                self.summary_writer.add_scalars('above_threshold_samples',
                                                {'mnist_train': mnist_train_above_threshold,
                                                 'mnist_test': mnist_test_above_threshold},
                                                self.epoch)

                self.summary_writer.flush()

            if self.epoch % save_every_n_epochs == save_every_n_epochs-1:
                self.save()


class GeneratorDiscriminatorTrainer:

    def __init__(self, device, params, encoder_classifier,
                 encoder_classifier_optimizer, svhn_trainsplit,
                 svhn_testsplit, mnist_trainsplit, mnist_testsplit, summary_writer=None,
                 experiment_name='generator_discriminator', ckpt_file=None, ckpt_root=os.path.join('ckpts', '')):

        self.device = device

        self.batch_size = params.batch_size
        self.ckpt_root = ckpt_root
        self.experiment_name = experiment_name
        self.summary_writer = summary_writer

        self.svhn_trainsplit_loader = generic_dataloader(self.device, svhn_trainsplit, shuffle=True,
                                                         batch_size=params.batch_size)
        self.svhn_testsplit_loader = generic_dataloader(self.device, svhn_testsplit, shuffle=False,
                                                        batch_size=params.batch_size)

        self.svhn_trainsplit_loader_notshuffled = generic_dataloader(self.device, svhn_trainsplit, shuffle=False,
                                                                     batch_size=params.batch_size)

        self.mnist_trainsplit = mnist_trainsplit
        self.mnist_trainsplit_loader = generic_dataloader(self.device, mnist_trainsplit, shuffle=False,
                                                          batch_size=params.batch_size)
        self.mnist_testsplit = mnist_testsplit
        self.mnist_testsplit_loader = generic_dataloader(self.device, mnist_testsplit, shuffle=False,
                                                         batch_size=params.batch_size)

        self.confidence_threshold = params.encoder_classifier_confidence_threshold

        self.encoder_classifier = encoder_classifier
        self.encoder_classifier_criterion = nn.CrossEntropyLoss()
        self.encoder_classifier_optimizer = encoder_classifier_optimizer

        self.generator = Generator(device, 3).to(device)
        self.generator_deception_criterion = nn.CrossEntropyLoss()
        self.generator_reconstruction_criterion = nn.MSELoss()
        self.generator_optimizer = optim.Adam(self.generator.parameters(),
                                              lr=params.generator_adam_lr,
                                              betas=(params.generator_adam_beta1, params.generator_adam_beta2))

        self.discriminator_svhn = Discriminator(3).to(device)
        self.discriminator_svhn_criterion = nn.CrossEntropyLoss()
        self.discriminator_svhn_optimizer = optim.Adam(self.discriminator_svhn.parameters(),
                                                       lr=params.discriminator_adam_lr,
                                                       betas=(params.discriminator_adam_beta1,
                                                              params.discriminator_adam_beta2))

        self.discriminator_mnist = Discriminator(3).to(device)
        self.discriminator_mnist_criterion = nn.CrossEntropyLoss()
        self.discriminator_mnist_optimizer = optim.Adam(self.discriminator_mnist.parameters(),
                                                        lr=params.discriminator_adam_lr,
                                                        betas=(params.discriminator_adam_beta1,
                                                               params.discriminator_adam_beta2))

        self.dupgan_alpha = params.dupgan_alpha
        self.dupgan_beta = params.dupgan_beta

        self.epoch = -1

        # allocate space for fake_labels for discriminators, don't copy the same [5,5,5,...] fake class labels
        # every time
        self.fake_label_pool = torch.full((self.batch_size,), 5, dtype=torch.long, device=self.device)

        if ckpt_file is not None:
            self.load(ckpt_file)
            mnist_hc_dataset, pseudolabels = self.get_high_confidence_mnist_dataset_with_pseudolabels()
            mnist_hc_loader = generic_dataloader(self.device, mnist_hc_dataset, shuffle=True,
                                                 batch_size=self.batch_size)
            self.log(mnist_hc_loader, pseudolabels)

    def save(self):
        torch.save({
            'model': 'generator_discriminator',
            'encoder_classifier_state_dict': self.encoder_classifier.state_dict(),
            'encoder_classifier_optimizer_state_dict': self.encoder_classifier_optimizer.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_mnist_state_dict': self.discriminator_mnist.state_dict(),
            'discriminator_mnist_optimizer_state_dict': self.discriminator_mnist_optimizer.state_dict(),
            'discriminator_svhn_state_dict': self.discriminator_svhn.state_dict(),
            'discriminator_svhn_optimizer_state_dict': self.discriminator_svhn_optimizer.state_dict(),
            'experiment_name': self.experiment_name,
            'epoch': self.epoch,
            'dupgan_alpha': self.dupgan_alpha,
            'dupgan_beta': self.dupgan_beta},
            os.path.join(self.ckpt_root, '%s_%d.tar' % (self.experiment_name, self.epoch)))

    def load(self, ckpt_file):
        checkpoint = torch.load(ckpt_file, map_location=self.device)
        self.encoder_classifier.load_state_dict(checkpoint['encoder_classifier_state_dict'])
        self.encoder_classifier_optimizer.load_state_dict(checkpoint['encoder_classifier_optimizer_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_mnist.load_state_dict(checkpoint['discriminator_mnist_state_dict'])
        self.discriminator_mnist_optimizer.load_state_dict(checkpoint['discriminator_mnist_optimizer_state_dict'])
        self.discriminator_svhn.load_state_dict(checkpoint['discriminator_svhn_state_dict'])
        self.discriminator_svhn_optimizer.load_state_dict(checkpoint['discriminator_svhn_optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.experiment_name = checkpoint['experiment_name']
        self.dupgan_alpha = checkpoint['dupgan_alpha']
        self.dupgan_beta = checkpoint['dupgan_beta']


    def get_high_confidence_mnist_dataset_with_pseudolabels(self):

        self.encoder_classifier.eval()
        with torch.no_grad():
            all_indices = torch.tensor([], dtype=int)
            all_pseudolabels = torch.tensor([], dtype=int)

            for data in self.mnist_trainsplit_loader:
                inputs, _, indices = data
                inputs = inputs.to(self.device)

                classifier_out, _ = self.encoder_classifier(inputs)

                confidence, pseudolabels = torch.max(nn.functional.softmax(classifier_out, 1), 1)
                confidence = confidence.squeeze()
                idx = confidence >= self.confidence_threshold

                pseudolabels = pseudolabels.squeeze()[idx].cpu()
                indices = indices[confidence >= self.confidence_threshold]

                all_pseudolabels = torch.cat((all_pseudolabels, pseudolabels), dim=0)
                all_indices = torch.cat((all_indices, indices), dim=0)

        self.encoder_classifier.train()

        dataset = SubsetWithIndices(self.mnist_trainsplit, all_indices)
        return (dataset, all_pseudolabels)

    # simple wrappers for model.xxxxx since there are so many models and optimizers
    def zero_grad(self):
        self.encoder_classifier_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()
        self.discriminator_svhn_optimizer.zero_grad()
        self.discriminator_mnist_optimizer.zero_grad()

    def eval(self):
        self.generator.eval()
        self.discriminator_svhn.eval()
        self.discriminator_mnist.eval()
        self.encoder_classifier.eval()

    def train(self):
        self.encoder_classifier.train()
        self.discriminator_mnist.train()
        self.discriminator_svhn.train()
        self.generator.train()

    # calculate important metrics in this function

    def evaluate(self, data_loader, domain_code, pseudolabels=None):

        # calculate some important metrics here, maybe log them later
        # domain code 0 -> svhn
        # domain code 1 -> mnist
        # domain code 2 -> mnist_high_confidence used in training, uses pseudolabels to calculate metrics

        self.eval()
        with torch.no_grad():

            total = 0

            encoder_classifier_correct = 0
            encoder_classifier_loss = 0

            discriminator_mnist_loss = 0
            discriminator_svhn_loss = 0
            discriminator_mnist_correct = 0
            discriminator_svhn_correct = 0

            generator_reconstruction_loss = 0
            generator_deception_loss = 0
            generator_deception_correct = 0


            for data in data_loader:

                # mnist_hc_indices will be empty for svhn, one item list for mnist, 2 item list for mnist high conf
                inputs, labels, *mnist_hc_indices = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device) if domain_code != 2 \
                    else pseudolabels[mnist_hc_indices[-1]].to(self.device)

                # determine which classifier should recieve which labels for loss
                discriminator_svhn_labels = labels if domain_code == 0 else self.fake_label_pool[:labels.shape[0]]
                discriminator_mnist_labels = labels if domain_code != 0 else self.fake_label_pool[:labels.shape[0]]

                # calculate everything

                classifier_out, latent_out = self.encoder_classifier(inputs)
                fake_svhn = self.generator(latent_out, 0)
                fake_mnist = self.generator(latent_out, 1)

                fake_of_the_same = fake_svhn if domain_code == 0 else fake_mnist

                discriminator_svhn_out = self.discriminator_svhn(inputs) if domain_code == 0 \
                    else self.discriminator_svhn(fake_svhn)
                discriminator_mnist_out = self.discriminator_mnist(fake_mnist) if domain_code == 0\
                    else self.discriminator_mnist(inputs)
                discriminator_opposite_out = discriminator_mnist_out if domain_code == 0 else discriminator_svhn_out

                # encoder classifier metrics

                encoder_classifier_loss += self.dupgan_beta*\
                                           self.encoder_classifier_criterion(classifier_out, labels).item()
                _, predicted = torch.max(classifier_out, 1)
                encoder_classifier_correct += (predicted == labels).sum().item()

                # discriminator_mnist metrics

                discriminator_mnist_loss += self.discriminator_mnist_criterion(discriminator_mnist_out,
                                                                               discriminator_mnist_labels)
                _, predicted = torch.max(discriminator_mnist_out, 1)
                discriminator_mnist_correct += (predicted == discriminator_mnist_labels).sum().item()

                # discriminator_svhn metrics

                discriminator_svhn_loss += self.discriminator_svhn_criterion(discriminator_svhn_out,
                                                                             discriminator_svhn_labels)
                _, predicted = torch.max(discriminator_svhn_out, 1)
                discriminator_svhn_correct += (predicted == discriminator_svhn_labels).sum().item()

                # generator metrics

                generator_reconstruction_loss += self.dupgan_alpha*\
                                                 self.generator_reconstruction_criterion(fake_of_the_same, inputs)
                generator_deception_loss += self.generator_deception_criterion(discriminator_opposite_out, labels)
                _, predicted = torch.max(discriminator_opposite_out, 1)
                discriminator_svhn_correct += (predicted == labels).sum().item()

                # count dataset

                total += labels.size(0)
        self.train()

        result_dict = {
            'encoder_classifier_accuracy': 100*encoder_classifier_correct/total,
            'encoder_classifier_loss': encoder_classifier_loss/total,
            'discriminator_mnist_loss': discriminator_mnist_loss/total,
            'discriminator_svhn_loss': discriminator_svhn_loss/total,
            'discriminator_mnist_accuracy': 100*discriminator_mnist_correct/total,
            'discriminator_svhn_accuracy': 100*discriminator_svhn_correct/total,
            'generator_reconstruction_loss': generator_reconstruction_loss/total,
            'generator_deception_loss': generator_deception_loss/total,
            'generator_loss': (generator_reconstruction_loss+generator_deception_loss)/total,
            'loss': (encoder_classifier_loss+discriminator_mnist_loss+discriminator_svhn_loss+
                     generator_reconstruction_loss+generator_deception_loss)/total,
            'generator_deception_accuracy': 100*generator_deception_correct/total,
        }

        return result_dict, total

    def reconstruct_images_from_dataset(self, dataloader, domain_code):

        # reconstructs from half the batch_size, takes from the second batches because they had more zeroes

        # domain code:
        # 0 to convert to svhn
        # 1 to convert to mnist

        dataiter = iter(dataloader)
        next(dataiter)
        inputs, *_ = next(dataiter)
        inputs = inputs.to(self.device)

        grid = torch.empty((self.batch_size*2, *inputs.shape[1:]), dtype=inputs.dtype, device=self.device)

        self.eval()
        with torch.no_grad():
            _, latent_out = self.encoder_classifier(inputs)
            fake = self.generator(latent_out, domain_code)
        self.train()

        grid[0::2, ...] = inputs
        grid[1::2, ...] = fake

        # pixel intensities were in range [-1, 1] they need to be converted back to [0 1]

        return make_grid(grid, nrow=16, padding=0, normalize=True, range=(-1, 1))


    def log_reconstructed_images_from_dataset(self, dataloader, dataset_name, domain_code):

        dataset_name = dataset_name+'_to_svhn' if domain_code == 0 else dataset_name+'_to_mnist'
        grid = self.reconstruct_images_from_dataset(dataloader, domain_code)
        self.summary_writer.add_image(dataset_name, grid, self.epoch)

    def log(self, mnist_hc_loader, pseudolabels):
        logging.info('Logging...')

        svhn_train_result_dict, svhn_train_total = self.evaluate(self.svhn_trainsplit_loader, 0)

        # uncomment if it is really required? I don't think so, just waste of time
        #svhn_test_result_dict, svhn_test_total = self.evaluate(self.svhn_testsplit_loader, 0)

        mnist_train_result_dict, mnist_train_total = self.evaluate(self.mnist_trainsplit_loader, 1)
        mnist_test_result_dict, mnist_test_total = self.evaluate(self.mnist_testsplit_loader, 1)
        mnist_hc_result_dict, mnist_hc_total = self.evaluate(mnist_hc_loader, 1, pseudolabels=pseudolabels)

        training_result_dict = {key: (svhn_train_result_dict[key]*svhn_train_total +
                                     mnist_hc_result_dict[key]*mnist_hc_total)/(svhn_train_total+mnist_hc_total)
                                for (key, _) in svhn_train_result_dict.items()}

        prepend_append_tag = lambda prep, app, dictt: {prep+key+app: value for (key, value) in dictt.items()}

        # datasets the model is not being trained on, these are for validation/test purposes
        other_dataset_tags = [(mnist_test_result_dict, 'mnist_test'),
                              # (svhn_test_result_dict, 'svhn_test'),
                              (mnist_train_result_dict, 'mnist_train')]

        # datasets the model is being trained on
        main_dataset_tags = [(svhn_train_result_dict, 'svhn_train'),
                             (mnist_hc_result_dict, 'mnist_hc'),
                             (training_result_dict, 'training_total')]

        for result_dict, tag in other_dataset_tags:
            result_dict = prepend_append_tag('other/', '/'+tag, result_dict)
            for (key, value) in result_dict.items():
                self.summary_writer.add_scalar(key, value, self.epoch)

        for result_dict, tag in main_dataset_tags:
            result_dict = prepend_append_tag('main/', '/'+tag, result_dict)
            for (key, value) in result_dict.items():
                self.summary_writer.add_scalar(key, value, self.epoch)

        self.log_reconstructed_images_from_dataset(self.svhn_trainsplit_loader_notshuffled, 'svhn_train', 0)
        self.log_reconstructed_images_from_dataset(self.mnist_trainsplit_loader, 'mnist_train', 1)
        self.log_reconstructed_images_from_dataset(self.svhn_trainsplit_loader_notshuffled, 'svhn_train', 1)

        # crucuial values to log (some of them are already logged but repeating would not hurt

        self.summary_writer.add_scalar('main/mnist_high_confidence_sample_count', mnist_hc_total, self.epoch)
        self.summary_writer.add_scalar('main/mnist_test_encoder_classifier_accuracy',
                            mnist_test_result_dict['encoder_classifier_accuracy'], self.epoch)

        self.summary_writer.flush()

        logging.info('Logging Over')

    def train_until_epoch(self, last_epoch, eval_every_n_epochs=1, save_every_n_epochs=1):

        # an epoch is defined as looping for the number of batches the largest dataset has here:
        # meaning svhn source dataset.

        for self.epoch in range(self.epoch+1, last_epoch):

            logging.info('generator_discriminator epoch: %d', self.epoch)

            # determine mnist_inputs with high confidence and use only them in training
            mnist_hc_dataset, pseudolabels = self.get_high_confidence_mnist_dataset_with_pseudolabels()
            mnist_hc_loader = generic_dataloader(self.device, mnist_hc_dataset, shuffle=True,
                                                 batch_size=self.batch_size)
            mnist_iterator = iter(mnist_hc_loader)
            logging.info('mnist_hc_dataset size: %d', len(mnist_hc_dataset))
            logging.info('mnist_iterator size: %d', len(mnist_iterator))

            mnist_ind = 0

            for svhn_data in self.svhn_trainsplit_loader:

                # since mnist is smaller need to use iterators to jointly go over the datasets

                # hack to skip 1 sample batches since they give an error with batchnorm
                while True:
                    if mnist_ind >= len(mnist_iterator):
                        logging.info('mnist batches recycled.')
                        mnist_iterator = iter(mnist_hc_loader)
                        mnist_ind = 0
                    mnist_hc_inputs, *_, mnist_hc_indices = next(mnist_iterator)
                    mnist_ind += 1
                    if mnist_hc_inputs.shape[0] > 1:
                        break

                mnist_hc_inputs = mnist_hc_inputs.to(self.device)
                mnist_hc_labels = pseudolabels[mnist_hc_indices].to(self.device)

                svhn_inputs, svhn_labels = svhn_data
                svhn_inputs = svhn_inputs.to(self.device)
                svhn_labels = svhn_labels.to(self.device)

                # Training below

                # Not putting generator/classifier on eval because
                # https://discuss.pytorch.org/t/why-dont-we-put-models-in-train-or-eval-modes-in-dcgan-example/7422/2

                # discriminator_mnist training
                self.zero_grad()

                _, svhn_latent_out = self.encoder_classifier(svhn_inputs)
                svhn_to_fake_mnist = self.generator(svhn_latent_out, 1)

                svhn_to_fake_mnist_discriminator_out = self.discriminator_mnist(svhn_to_fake_mnist)
                real_mnist_discriminator_out = self.discriminator_mnist(mnist_hc_inputs)

                discriminator_mnist_fake_loss = self.discriminator_mnist_criterion(svhn_to_fake_mnist_discriminator_out,
                                                 self.fake_label_pool[:svhn_to_fake_mnist_discriminator_out.shape[0]])

                discriminator_mnist_real_loss = self.discriminator_mnist_criterion(real_mnist_discriminator_out,
                                                                                   mnist_hc_labels)

                discriminator_mnist_loss = discriminator_mnist_fake_loss + discriminator_mnist_real_loss

                discriminator_mnist_loss.backward()
                self.discriminator_mnist_optimizer.step()

                # discriminator_svhn_training
                self.zero_grad()

                _, mnist_hc_latent_out = self.encoder_classifier(mnist_hc_inputs)
                mnist_to_fake_svhn = self.generator(mnist_hc_latent_out, 0)

                mnist_to_fake_svhn_discriminator_out = self.discriminator_svhn(mnist_to_fake_svhn)
                real_svhn_discriminator_out = self.discriminator_svhn(svhn_inputs)

                discriminator_svhn_fake_loss = self.discriminator_svhn_criterion(mnist_to_fake_svhn_discriminator_out,
                                                self.fake_label_pool[:mnist_to_fake_svhn_discriminator_out.shape[0]])

                discriminator_svhn_real_loss = self.discriminator_svhn_criterion(real_svhn_discriminator_out,
                                                                                 svhn_labels)

                discriminator_svhn_loss = discriminator_svhn_fake_loss + discriminator_svhn_real_loss

                discriminator_svhn_loss.backward()
                self.discriminator_svhn_optimizer.step()

                # generator and encoder_classifier training
                self.zero_grad()

                mnist_hc_classifier_out, mnist_hc_latent_out = self.encoder_classifier(mnist_hc_inputs)
                svhn_classifier_out, svhn_latent_out = self.encoder_classifier(svhn_inputs)

                svhn_to_fake_svhn = self.generator(svhn_latent_out, 0)
                svhn_to_fake_mnist = self.generator(svhn_latent_out, 1)
                mnist_to_fake_svhn = self.generator(mnist_hc_latent_out, 0)
                mnist_to_fake_mnist = self.generator(mnist_hc_latent_out, 1)

                svhn_to_fake_mnist_discriminator_out = self.discriminator_mnist(svhn_to_fake_mnist)

                mnist_to_fake_svhn_discriminator_out = self.discriminator_svhn(mnist_to_fake_svhn)

                # generator losses

                deceive_discriminator_mnist_loss = self.generator_deception_criterion(
                    svhn_to_fake_mnist_discriminator_out, svhn_labels)

                deceive_discriminator_svhn_loss = self.generator_deception_criterion(
                    mnist_to_fake_svhn_discriminator_out, mnist_hc_labels)

                reconstruction_mnist_loss = self.generator_reconstruction_criterion(mnist_to_fake_mnist,
                                                                                    mnist_hc_inputs)
                reconstruction_svhn_loss = self.generator_reconstruction_criterion(svhn_to_fake_svhn,
                                                                                   svhn_inputs)
                generator_loss = deceive_discriminator_mnist_loss +\
                    deceive_discriminator_svhn_loss +\
                    self.dupgan_alpha*(reconstruction_mnist_loss+reconstruction_svhn_loss)

                # classifier losses

                mnist_classification_loss = self.encoder_classifier_criterion(mnist_hc_classifier_out, mnist_hc_labels)
                svhn_classification_loss = self.encoder_classifier_criterion(svhn_classifier_out, svhn_labels)

                classification_loss = self.dupgan_beta*(mnist_classification_loss + svhn_classification_loss)

                generator_ec_loss = generator_loss + classification_loss
                generator_ec_loss.backward()

                self.generator_optimizer.step()
                self.encoder_classifier_optimizer.step()

            # logging and saving stuff
            if self.epoch % eval_every_n_epochs == eval_every_n_epochs-1:
                self.log(mnist_hc_loader, pseudolabels)

            if self.epoch % save_every_n_epochs == save_every_n_epochs-1:
                self.save()
