#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import torch.distributions as tdist
import scipy.io
import itertools


class NbrReg(torch.nn.Module):
    def __init__(self, lex_size, bit_size=32, h_size=1000):
        super(NbrReg, self).__init__()
        self.lnr_h1 = torch.nn.Linear(lex_size, h_size)
        self.lnr_h2 = torch.nn.Linear(h_size, h_size)
        self.lnr_mu = torch.nn.Linear(h_size, bit_size)
        self.lnr_sigma = torch.nn.Linear(h_size, bit_size)
        self.lnr_rec_doc = torch.nn.Linear(bit_size, lex_size)
        self.lnr_nn_rec_doc = torch.nn.Linear(bit_size, lex_size)

    def forward(self, docs):
        mu, sigma = self.encode(docs)
        qdist = tdist.Normal(mu, sigma)
        log_prob_words, log_nn_prob_words = self.decode(qdist.rsample())
        return qdist, log_prob_words, log_nn_prob_words

    def encode(self, docs):
        relu = torch.nn.ReLU()
        sigmoid = torch.nn.Sigmoid()
        hidden = relu(self.lnr_h2(relu(self.lnr_h1(docs))))
        mu = self.lnr_mu(hidden)
        # Use sigmoid for positive standard deviation
        sigma = sigmoid(self.lnr_sigma(hidden))
        return mu, sigma

    def decode(self, latent):
        # Listening to the advice on the torch.nn.Softmax; we use
        # LogSoftmax since we'll use NLLLoss. Rather than
        # multiplication of each word prob and then taking log, we
        # compute log and then sum the probabilities.
        log_softmax = torch.nn.LogSoftmax(dim=1)
        log_prob_words = log_softmax(self.lnr_rec_doc(latent))
        log_nn_prob_words = log_softmax(self.lnr_nn_rec_doc(latent))
        return log_prob_words, log_nn_prob_words


def doc_rec_loss(log_prob_words, doc_batch):
    doc_mask = doc_batch.clone()
    doc_mask[torch.where(doc_mask != 0)] = 1.0
    rel_log_prob_words = torch.mul(log_prob_words, doc_mask)
    return -torch.mean(torch.sum(rel_log_prob_words, dim=1))


def doc_nn_rec_loss(log_nn_prob_words, knn_batch, train_docs):
    word_mask = None
    for knn in knn_batch:
        nn_docs = torch.from_numpy(train_docs[knn].todense())
        nn_mask = torch.sum(nn_docs != 0, dim=0) != 0
        nn_mask = nn_mask.reshape(1, len(nn_mask))
        if word_mask is None:
            word_mask = nn_mask
        else:
            word_mask = torch.cat((word_mask, nn_mask))
    rel_log_nn_prob_words = torch.mul(log_nn_prob_words, word_mask)
    return -torch.mean(torch.sum(rel_log_nn_prob_words, dim=1))


def binarize(means, threshold):
    hashes = means.clone()
    ones_i = hashes > threshold
    zeros_i = hashes <= threshold
    hashes[ones_i] = 1
    hashes[zeros_i] = 0
    return hashes


def encode_with_batches(docs, model, bsize):
    num_iter = int(np.ceil(docs.shape[0] / bsize))
    means = None
    for i in range(num_iter):
        batch = docs[i * bsize:(i+1) * bsize].todense()
        batch = torch.from_numpy(batch).double()
        mu, _ = model.encode(batch)
        if means is None:
            means = mu
        else:
            means = torch.cat((means, mu))
    return means


def hamming_score(test, train):
    return torch.sum(test == train, dim=1)


def test(train_docs, train_cats, test_docs, test_cats, model, bsize=100,
         k=100):
    model.eval()
    with torch.no_grad():
        train_means = encode_with_batches(train_docs, model, bsize)
        test_means = encode_with_batches(test_docs, model, bsize)
        threshold = torch.median(train_means, dim=0).values
        train_hash = binarize(train_means, threshold)
        test_hash = binarize(test_means, threshold)
        prec_sum = 0.0
        for i, th in enumerate(test_hash):
            hd = hamming_score(th.repeat(train_hash.shape[0], 1),
                               train_hash)
            _, topk_i = torch.topk(hd, k)
            rel = 0
            rel_cat = torch.from_numpy(test_cats[i].todense())
            for di in topk_i:
                train_cat = torch.from_numpy(train_cats[di].todense())
                rel += torch.sum(torch.mul(train_cat, rel_cat)).item()

            prec_sum += rel / k
        return prec_sum / test_hash.shape[0]


def train(train_docs, train_cats, train_knn, cv_docs, cv_cats, bitsize=32,
          epoch=30, bsize=100, lr=1e-3, latent_size=1000, resume=None,
          imp_trial=0):
    nsize, lexsize = train_docs.shape
    num_iter = int(np.ceil(nsize / bsize))
    model = resume if resume else NbrReg(lexsize, bitsize, h_size=latent_size)
    model.double()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    norm = tdist.Normal(0, 1)
    best_prec = 0.0
    trial = 0
    epoch_range = itertools.count() if imp_trial else epoch
    epoch = "INF" if imp_trial else epoch

    for e in epoch_range:
        model.train()
        losses = []
        for i in range(num_iter):
            print(f"Epoch: {e + 1}/{epoch}, Iteration: {i + 1}/{num_iter}",
                  end="\r")
            batch_i = np.random.choice(nsize, bsize)
            np_batch = train_docs[batch_i].todense()
            doc_batch = torch.from_numpy(np_batch).double()
            knn_batch = train_knn[batch_i]
            optim.zero_grad()
            qdist, log_prob_words, log_nn_prob_words = model(doc_batch)
            doc_rl = doc_rec_loss(log_prob_words, doc_batch)
            doc_nn_rl = doc_nn_rec_loss(log_nn_prob_words, knn_batch,
                                        train_docs)
            kl_loss = tdist.kl_divergence(qdist, norm)
            kl_loss = torch.mean(torch.sum(kl_loss, dim=1))
            loss = doc_rl + doc_nn_rl + kl_loss
            losses.append(loss.item())
            loss.backward()
            optim.step()
        avg_loss = np.mean(losses)
        avg_prec = test(train_docs, train_cats, cv_docs, cv_cats, model)
        best_prec = max(avg_prec, best_prec)
        print(f"Epoch {e + 1}: Avg Loss: {avg_loss}, Avg Prec: {avg_prec}")
        if best_prec == avg_prec:
            trial = 0
        else:
            trial += 1
            if trial == imp_trial:
                print(f"Avg Prec could not be improved for {imp_trial} times, "
                      "giving up training")
                break

    return model, best_prec


def load_model(mpath, lexsize, bit_size, h_size):
    model = NbrReg(lexsize, bit_size=bit_size, h_size=h_size)
    model.double()
    model.load_state_dict(torch.load(mpath))
    return model


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Subcommand help", dest="task")
    train_parser = subparsers.add_parser("train",
                                         help="Train a model and save")
    subparsers.add_parser("test", help="Load and test a model")

    parser.add_argument("data", help="Input data")
    parser.add_argument("model", help="Path to model")
    parser.add_argument("-b", "--bit-size", type=int, default=32)
    parser.add_argument("-t", "--latent-size", type=int, default=1000)

    train_parser.add_argument("-e", "--epoch", type=int, default=15)
    train_parser.add_argument("-a", "--batch-size", type=int, default=100)
    train_parser.add_argument("-l", "--learning-rate", type=float,
                              default=1e-3)
    train_parser.add_argument("-k", "--knn-size", type=int, default=20)
    train_parser.add_argument("-r", "--resume",
                              help="Resume training the given model")
    train_parser.add_argument("-i", "--trial", type=int, default=0,
                              help="Give up training if no improvements "
                              "have been observed for given number of epochs")

    args = parser.parse_args()

    data = scipy.io.loadmat(args.data)
    train_docs = data["train"]
    train_cats = data["gnd_train"]
    if args.task == "train":
        cv_docs = data["cv"]
        cv_cats = data["gnd_cv"]
        train_knn = data["train_knn"]
        if args.resume:
            res_m = load_model(args.resume, train_docs.shape[1], args.bit_size,
                               args.latent_size)
        else:
            res_m = None

        model, best_cv_prec = train(train_docs, train_cats,
                                    train_knn[:, :args.knn_size], cv_docs,
                                    cv_cats, bitsize=args.bit_size,
                                    epoch=args.epoch, lr=args.learning_rate,
                                    latent_size=args.latent_size, resume=res_m,
                                    imp_trial=args.trial)
        torch.save(model.state_dict(), args.model)
    else:
        test_docs = data["test"]
        test_cats = data["gnd_test"]
        model = load_model(args.model, train_docs.shape[1], args.bit_size,
                           args.latent_size)
        avg_prec = test(train_docs, train_cats, test_docs, test_cats, model)
        print(f"Average test precision: {avg_prec}")
    return 0


if __name__ == "__main__":
    exit(main())
