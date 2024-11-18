import os
import json
from argparse import ArgumentParser
from tasks import TASKS



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--task", type=str, default="comparison", choices=TASKS.keys())
    parser.add_argument("--output_dir", type=str, default="outputs_task")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(args)
    task = TASKS[args.task](args.model, args.output_dir, args.debug)
    print(task.task_name)
    score = task.run_pipeline()
    print(f"Task: {args.task}, Model: {args.model}, Score: {score}")