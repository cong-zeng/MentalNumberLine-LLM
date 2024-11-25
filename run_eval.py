import os
import json
import time
from argparse import ArgumentParser
import numpy as np

from EmbeddingGenerator import EmbeddingGenerator

def eval(args):
    model = args.model
    context = args.context
    num_type = args.num_type
    output_dir = args.output_dir
    token_method = args.token_method
    results_dir = args.results_dir
    is_probing = args.is_probing
    print(f"model:{model}, context:{context}, num_type:{num_type}")
    embedding_generator = EmbeddingGenerator(model, context, num_type, output_dir, token_method, results_dir, is_probing)
    embedding = embedding_generator.run_pipeline()
    print(np.array(embedding).shape)
    embedding_generator.evaluate()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-2")
    parser.add_argument("--context", type=str, default="nocontext")
    parser.add_argument("--num_type", type=str, default="single", choices=["single", "sequence"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--token_method", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--is_probing", action="store_true", default=False)
    args = parser.parse_args()
    eval(args)