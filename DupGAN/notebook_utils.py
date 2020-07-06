from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import numpy as np

from utils import generic_dataloader

encoder_classifier_evaluations = None
generator_discriminator_evaluations = None


def display_table(tabldict):
    content = "<table><tr>"+"".join([f"<th>{key}</th>" for key in tabldict])+"</tr>"
    for i in range(len(tabldict[next(iter(tabldict))])):
        content += "<tr>"
        for key in tabldict:
            form = f"{tabldict[key][i]:.4f}" if type(tabldict[key][i]) == float else str(tabldict[key][i])
            content = content + "<td>"+form+"</td>"
        content += "</tr>"
    content += "</table>"

    display(HTML(content))


def encoder_classifier_notebook_evaluator_evaluate_and_print(self):
    svhn_train_loss, svhn_train_accuracy, *_ = self.evaluate(self.svhn_trainsplit_loader)
    _, svhn_test_accuracy, *_ = self.evaluate(self.svhn_testsplit_loader)

    _, mnist_train_accuracy, _, mnist_train_above_threshold_accuracy = self.evaluate(self.mnist_trainsplit_loader)

    results = {"Epoch": self.epoch,
               "SVHN Train Split Loss": svhn_train_loss,
               "SVHN Train Split Classification Acc. (%)": svhn_train_accuracy,
               "SVHN Test Split Classification Acc. (%)": svhn_test_accuracy,
               "MNIST Train Split Above Threshold (%)": mnist_train_above_threshold_accuracy}

    for key in results:
        encoder_classifier_evaluations[key].append(results[key])

    # clear output before printing same table with more rows
    clear_output(wait=True)
    display_table(encoder_classifier_evaluations)


def encoder_classifier_notebook_evaluator_reset():
    global encoder_classifier_evaluations
    encoder_classifier_evaluations = {"Epoch": [],
                                      "SVHN Train Split Loss": [],
                                      "SVHN Train Split Classification Acc. (%)": [],
                                      "SVHN Test Split Classification Acc. (%)": [],
                                      "MNIST Train Split Above Threshold (%)": []}


def generator_discriminator_notebook_evaluator_reset():
    global generator_discriminator_evaluations
    generator_discriminator_evaluations = {"Epoch": [],
                                           "Discriminator Loss": [],
                                           "Generator Deception Loss": [],
                                           "Generator Reconstruction Loss": [],
                                           "Generator Loss": [],
                                           "MNIST High Conf. Sample Count": []}


def generator_discriminator_notebook_evaluator_evaluate_and_print(self, mnist_hc_loader, pseudolabels):

    svhn_train_result_dict, svhn_train_total = self.evaluate(self.svhn_trainsplit_loader, 0)
    mnist_hc_result_dict, mnist_hc_total = self.evaluate(mnist_hc_loader, 1, pseudolabels=pseudolabels)
    training_result_dict = {key: (svhn_train_result_dict[key] * svhn_train_total +
                                  mnist_hc_result_dict[key] * mnist_hc_total) / (svhn_train_total + mnist_hc_total)
                            for (key, _) in svhn_train_result_dict.items()}
    results = {"Epoch": self.epoch,
               "Discriminator Loss": training_result_dict["discriminator_mnist_loss"].item() +
                                     training_result_dict["discriminator_svhn_loss"].item(),
               "Generator Deception Loss": training_result_dict["generator_deception_loss"].item(),
               "Generator Reconstruction Loss": training_result_dict["generator_reconstruction_loss"].item(),
               "Generator Loss": training_result_dict["generator_loss"].item(),
               "MNIST High Conf. Sample Count": mnist_hc_total}

    for key in results:
        generator_discriminator_evaluations[key].append(results[key])

    clear_output(wait=True)
    img = self.reconstruct_images_from_dataset(self.svhn_trainsplit_loader_notshuffled, 1)
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

    display_table(generator_discriminator_evaluations)


def generator_discriminator_notebook_evaluator_evaluate_goals_and_print(self):
    mnist_hc_dataset, pseudolabels = self.get_high_confidence_mnist_dataset_with_pseudolabels()
    mnist_hc_loader = generic_dataloader(self.device, mnist_hc_dataset, shuffle=True, batch_size=self.batch_size)
    generator_discriminator_notebook_evaluator_reset()
    generator_discriminator_notebook_evaluator_evaluate_and_print(self, mnist_hc_loader, pseudolabels)
    mnist_test_result_dict, _ = self.evaluate(self.mnist_testsplit_loader, 1)
    results = {"MNIST Test Split Final Classification Accuracy":
                   [mnist_test_result_dict['encoder_classifier_accuracy']]}
    display_table(results)


def generator_discriminator_notebook_load_and_print_params(self, ckpt_file):
    self.load(ckpt_file)
    self.load(ckpt_file)
    print("Discriminator ADAM learning rate:", self.discriminator_mnist_optimizer.param_groups[0]['lr'])
    print("Discriminator ADAM betas:", self.discriminator_mnist_optimizer.param_groups[0]['betas'])
    print("Generator ADAM learning rate:", self.generator_optimizer.param_groups[0]['lr'])
    print("Generator ADAM betas:", self.generator_optimizer.param_groups[0]['betas'])
    print("Encoder-Classifier ADAM learning rate:", self.encoder_classifier_optimizer.param_groups[0]['lr'])
    print("Encoder-Classifier ADAM betas:", self.encoder_classifier_optimizer.param_groups[0]['betas'])
    print("DUPGAN Alpha:", self.dupgan_alpha)
    print("DUPGAN Beta:", self.dupgan_beta)
    print("epoch:", self.epoch)
    print("experiment_name:", self.experiment_name)


def encoder_classifier_notebook_load_and_print_params(self, ckpt_file):
    self.load(ckpt_file)
    print("ADAM learning rate:", self.optimizer.param_groups[0]['lr'])
    print("ADAM betas:", self.optimizer.param_groups[0]['betas'])
    print("epoch:", self.epoch)
    print("experiment_name:", self.experiment_name)

