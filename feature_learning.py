import os
import argparse
import itertools

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

import cornac
from cornac.utils import cache
from cornac.data import TextModality, GraphModality
from cornac.data.reader import Reader, read_text
from cornac.data.text import BaseTokenizer
from cornac.models.vaecf.vaecf import VAE


parser = argparse.ArgumentParser(description="Feature learning with VAE")
parser.add_argument("-d", "--dataset", type=str, default="office",
                    choices=["office", "clothing", "sports"],
                    help="name of the dataset")
parser.add_argument("-w", "--which", type=str, default="user", 
                    choices=["user", "item"],
                    help="learning feature for user/item")
parser.add_argument("-k", "--latent_dim", type=int, default=20, 
                    help="number of the latent dimensions")
parser.add_argument("-e", "--encoder", type=str, default="[100]", 
                    help="structure of the encoder")
parser.add_argument("-a", "--act_fn", type=str, default="tanh",
                    choices=["sigmoid", "tanh", "relu", "relu6", "elu"],
                    help="non-linear activation function for the encoders")
parser.add_argument("-l", "--likelihood", type=str, default="pois",
                    choices=["pois", "bern", "gaus", "mult"],
                    help="likelihood function to fit the observations")
parser.add_argument("-ne", "--num_epochs", type=int, default=100, 
                    help="number of training epochs")
parser.add_argument("-bs", "--batch_size", type=int, default=128, 
                    help="batch size for training")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                    help="learning rate for training")
parser.add_argument("-kl", "--beta_kl", type=float, default=1.0,
                    help="beta weighting for the KL divergence")
parser.add_argument("-s", "--random_seed", type=int, default=123, 
                    help="random seed value")
parser.add_argument("-v", "--verbose", action="store_true", 
                    help="increase output verbosity")
args = parser.parse_args()
print(args)


class TextDataset(Dataset):
    def __init__(self, text_file):
        self.texts, self.ids = read_text(text_file, sep="\t")
        id_map = {_id: idx for idx, _id in enumerate(self.ids)}
        text_modality = TextModality(
            corpus=self.texts,
            ids=self.ids,
            tokenizer=BaseTokenizer(sep=" ", stop_words="english"),
            max_vocab=8000,
            max_doc_freq=0.5,
        ).build(id_map=id_map)
        self.bow_mat = text_modality.count_matrix
        self.bow_mat.data = np.ones_like(self.bow_mat.data, dtype=np.float32)
        self.input_dim = self.bow_mat.shape[1]

    def __len__(self):
        return self.bow_mat.shape[0]

    def __getitem__(self, idx):
        return idx, self.bow_mat[idx].A.ravel()


class ContextDataset(Dataset):
    def __init__(self, ctx_file):
        contexts = Reader().read(ctx_file, fmt="UI", sep="\t")
        self.ids = list(set(c[0] for c in contexts) | (set(c[1] for c in contexts)))
        id_map = {_id: idx for idx, _id in enumerate(self.ids)}
        graph_modality = GraphModality(data=contexts).build(id_map=id_map)
        self.ctx_mat = graph_modality.matrix
        self.ctx_mat.data = np.ones_like(self.ctx_mat.data, dtype=np.float32)
        self.input_dim = self.ctx_mat.shape[1]

    def __len__(self):
        return self.ctx_mat.shape[0]

    def __getitem__(self, idx):
        return idx, self.ctx_mat[idx].A.ravel()


def retrieve_dataset():
    dataset_dir = f"./data/{args.dataset}"
    os.makedirs(dataset_dir, exist_ok=True)

    train_path = cache(
        url=f"https://static.preferred.ai/bi-vae/{args.dataset}/train.txt",
        cache_dir=dataset_dir,
    )
    test_path = cache(
        url=f"https://static.preferred.ai/bi-vae/{args.dataset}/test.txt",
        cache_dir=dataset_dir,
    )
    reader = Reader()
    train_data = reader.read(fpath=train_path, sep="\t")
    test_data = reader.read(fpath=test_path, sep="\t")

    if args.which == "user":
        user_text_path = cache(
            url=f"https://static.preferred.ai/bi-vae/{args.dataset}/user_texts.txt",
            cache_dir=dataset_dir,
        )
        dataset = TextDataset(user_text_path)
        rating_ids = [x[0] for x in itertools.chain(train_data, test_data)]
    else:
        item_context_path = cache(
            url=f"https://static.preferred.ai/bi-vae/{args.dataset}/item_contexts.txt",
            cache_dir=dataset_dir,
        )
        dataset = ContextDataset(item_context_path)
        rating_ids = [x[1] for x in itertools.chain(train_data, test_data)]

    return dataset, rating_ids


if __name__ == "__main__":
    if args.random_seed:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)

    dataset, rating_ids = retrieve_dataset()
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    dtype = torch.float32
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    vae = VAE(
        z_dim=args.latent_dim,
        ae_structure=[dataset.input_dim] + eval(args.encoder),
        act_fn=args.act_fn,
        likelihood=args.likelihood,
    ).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    output = np.empty([len(dataset), args.latent_dim], dtype=np.float32)

    pbar = trange(1, args.num_epochs + 1, disable=not args.verbose)
    for _ in pbar:
        sum_loss = 0.0
        for batch_idx, batch_x in dataloader:
            batch_x = batch_x.to(device)
            rec_batch_x, mu, logvar = vae(batch_x)

            loss = vae.loss(batch_x, rec_batch_x, mu, logvar, args.beta_kl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            output[batch_idx.numpy()] = mu.detach().cpu().numpy()
        pbar.set_postfix(loss=(sum_loss / len(dataset)))

    # zeros for user/item without any information
    feature_ids = dataset.ids
    for _id in set(rating_ids) - set(feature_ids):
        feature_ids.append(_id)
        output = np.vstack([output, np.zeros([1, args.latent_dim], dtype=np.float32)])

    outfile = f"./data/{args.dataset}/{args.which}_features.npz"
    np.savez(outfile, ids=feature_ids, features=output)
    print(f"Output saved: {outfile}")