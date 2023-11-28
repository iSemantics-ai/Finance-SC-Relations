import argparse
from .config.core import config, TRAINED_MODEL_DIR, DATASET_DIR

def get_args():
    parser = argparse.ArgumentParser(description='SupplyChainClassifier')
    parser.add_argument('-s', '--request', help='Request like (train, eval, predict)', type=str, required=False, default = "train")
    parser.add_argument('-base', '--base-model', help='Which model to use', type=str, required=False, default="nlpaueb/sec-bert-base")
    parser.add_argument('-name', '--model-name', help='Name to be saved by', type=str, required=False)

    parser.add_argument('-ep', '--epochs', help='Number of epochs', type=int, required=False, default=2)


    # Augmentation Filter
    parser.add_argument('-st', '--sim-threshold', help='Similarity threshold to search for ', type=float, required=False, default=0.6)
    parser.add_argument('-mc', '--min-confident', help='Minimum Confident To Accept any record', type=float, required=False, default=0.5)
    parser.add_argument('-el', '--inverse-limit', help='The threshold that inverse label if the model reach', type=float, required=False, default=0.95)
    parser.add_argument('-tk', '--top-k', help='maximum number similarities to return', type=int, required=False, default=50)
    parser.add_argument('-sim', '--simcse', help='The SimCSE model to be loader', type=str, required=False, default="princeton-nlp/sup-simcse-roberta-large")
    parser.add_argument('-simemb', '--simcse-emb', help='Pre computed embeddings for similarity searching', type=str, required=False, default="roberta_large_production_embeddings.pt")


    # Training hyper-parameters

    parser.add_argument('-bs', '--batch-size', help='Batch size', type=int, required=False, default=32)
    parser.add_argument('-nw', '--num-workers', help='Number of CPU workers', type=int, required=False, default=0)
    parser.add_argument('-lr', '--learning-rate', help='Learning rate', type=float, required=False, default=1e-5)
    parser.add_argument('-wd', '--weight-decay', help='Weight decay', type=float, required=False, default=0.001)
    
    parser.add_argument('-ws', '--warmup-smooth', help='warmup smooth(lower = smoother)', type=float, required=False, default=0.8)
    parser.add_argument('-tr', '--truncate', help='Truncate the sequence length to', type=int, required=False, default=512)
    parser.add_argument('-wandb', '--wandb', help='Monitor Training with WandB', type=bool, required=False, default=False)
    parser.add_argument('-tf', '--train-file', help='The training file', type=str, required=False, default=config.app_config.training_data_file)
   
    parser.add_argument('-pa', '--patience', help='Patience to stop training', type=int, required=False, default=5)
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')
    parser.add_argument('-pr', '--project', help='the project name', type=str, required=False, default="SupplyChainClassifier")
    parser.add_argument('-cl', '--clip', help='Use clip to gradients', action='store_true')
    parser.add_argument('-fr', '--freeze', help='Freeze the embedding layer or not to use less GPU memory', type=bool, required=False, default=False)
    parser.add_argument('-lw', '--loss-weights', help='Weights for all losses', nargs='+', type=float, required=False, default=[1, 1, 1, 1])
    parser.add_argument('-sc', '--scheduler', help='Use scheduler to optimizer', action='store_true')
    parser.add_argument('-se', '--seed', help='Random seed', type=int, required=False, default=1)
    # Transformers
    parser.add_argument('-ad', '--attention-dropout', help='transformer attention dropout', type=float, required=False, default=0.1)
    parser.add_argument('-hd', '--hidden-dropout', help='transformer hidden dropout', type=float, required=False, default=0.1)

    args = vars(parser.parse_args())
    return args
args = get_args() 