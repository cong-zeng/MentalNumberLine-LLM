#!/usr/bin/env bash
# setup the environment
echo `date`, Setup the environment ...

model="llama3.1-8b mistral-7B Phi3.5-4b Qwen2.5-7B gemma2-9b"
# task="comparison classifyNum left_digit"

# task="classifyNum"
# task="comparison"
# models="Phi3.5-4b Qwen2.5-7B gemma2-9b"
models="Qwen2.5-7B"
task="left_digit"
for c in $task; do
    for m in $model; do
        echo `date`, Evaluating $m with task: $c ...
        python run_task_eval.py --model $m --task $c 
    done
done
  