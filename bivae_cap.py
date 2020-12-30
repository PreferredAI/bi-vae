import os
import argparse

import numpy as np
import torch
import cornac
from cornac.utils import cache
from cornac.data import FeatureModality


parser = argparse.ArgumentParser(description="BiVAE model with Constrained Adaptive Priors (CAP)")
parser.add_argument("-d", "--dataset", type=str, default="office",
                    choices=["office", "clothing", "sports", "epinions"],
                    help="name of the dataset")
parser.add_argument("-uc", "--user_cap", action="store_true", 
                    help="use CAP for the user side")
parser.add_argument("-ic", "--item_cap", action="store_true", 
                    help="use CAP for the item side")
parser.add_argument("-k", "--latent_dim", type=int, default=20, 
                    help="number of the latent dimensions")
parser.add_argument("-e", "--encoder", type=str, default="[40]",
                    help="structure of the user/item encoders")
parser.add_argument("-a", "--act_fn", type=str, default="tanh",
                    choices=["sigmoid", "tanh", "relu", "relu6", "elu"],
                    help="non-linear activation function for the encoders")
parser.add_argument("-l", "--likelihood", type=str, default="pois",
                    choices=["pois", "bern", "gaus"],
                    help="likelihood function to fit the rating observations")
parser.add_argument("-ne", "--num_epochs", type=int, default=500, 
                    help="number of training epochs")
parser.add_argument("-bs", "--batch_size", type=int, default=128, 
                    help="batch size for training")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                    help="learning rate for training")
parser.add_argument("-kl", "--beta_kl", type=float, default=1.0,
                    help="beta weighting for the KL divergence")
parser.add_argument("-tk", "--top_k", type=int, default=50, 
                    help="k cut-off for top-k evaluation")
parser.add_argument("-s", "--random_seed", type=int, default=123, 
                    help="random seed value")
parser.add_argument("-v", "--verbose", action="store_true", 
                    help="increase output verbosity")
args = parser.parse_args()
print(args)


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

    reader = cornac.data.Reader()
    train_data = reader.read(fpath=train_path, sep="\t")
    test_data = reader.read(fpath=test_path, sep="\t")

    user_features = np.load(f"{dataset_dir}/user_features.npz")
    item_features = np.load(f"{dataset_dir}/item_features.npz")

    return train_data, test_data, user_features, item_features


if __name__ == "__main__":
    train_data, test_data, user_features, item_features = retrieve_dataset()
    
    uf_modality = FeatureModality(user_features["features"], user_features["ids"])
    if_modality = FeatureModality(item_features["features"], item_features["ids"])
    
    eval_method = cornac.eval_methods.BaseMethod.from_splits(
        train_data=train_data,
        test_data=test_data,
        user_feature=uf_modality,
        item_feature=if_modality,
        seed=args.random_seed,
        verbose=args.verbose,
    )
    
    bivae = cornac.models.BiVAECF(
        k=args.latent_dim,
        encoder_structure=eval(args.encoder),
        act_fn=args.act_fn,
        likelihood=args.likelihood,
        n_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_kl=args.beta_kl,
        seed=args.random_seed,
        use_gpu=torch.cuda.is_available(),
        verbose=args.verbose,
        cap_priors={"user": args.user_cap, "item": args.item_cap}
    )

    topk_metrics = [cornac.metrics.NDCG(args.top_k), cornac.metrics.Recall(args.top_k)]

    cornac.Experiment(
        eval_method=eval_method, models=[bivae], metrics=topk_metrics, user_based=True
    ).run()
