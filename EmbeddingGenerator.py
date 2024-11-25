import os
import torch
import pandas as pd
import numpy as np
import json
import re
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset
from metrics import nearest_neighbor_analysis
from metrics import distance_linear_analysis
from metrics import metric_left_digit_effect
from probe_comparison import probe_num_compare_accu
from probe_addition import probe_num_addition_accu

model_fullnames = {  'gpt2': 'openai-community/gpt2',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'opt-13b': 'facebook/opt-13b',
                     'math_gpt2': 'Sharathhebbar24/math_gpt2',
                     'mathstral-7B': 'mistralai/Mathstral-7B-v0.1',
                     'opt-1.3b':"facebook/opt-iml-1.3b",
                     'llama3-8b':'meta-llama/Meta-Llama-3-8B-Instruct',
                     'gemma2-9b':"google/gemma-2-9b-it",
                     'Phi3.5-4b':"microsoft/Phi-3.5-mini-instruct",
                     'mistral-7B':"mistralai/Mistral-7B-Instruct-v0.3",
                     'llama3.1-8b':'meta-llama/Llama-3.1-8B-Instruct',
                     'Qwen2.5-7B': "Qwen/Qwen2.5-7B-Instruct",
                     'llama2-7b':"meta-llama/Llama-2-7b-chat-hf"
                     }

class EmbeddingGenerator():

    data_dir = '/home/zc/mml/dataset'

    def __init__(self, model, context, num_type_name, output_dir, token_method, results_dir, is_probing=False):
        self.prompt_data_name = context+'_'+num_type_name
        self.num_type_name = num_type_name
        self.model = model
        self.method = token_method
        self.data_df = self.read_prompt_data(self.prompt_data_name)
        self.embedding_dir = os.path.join(output_dir, 'embeddings')
        self.embedding = None
        self.results_dir = os.path.join(output_dir, results_dir)
        self.is_probing = is_probing

        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def read_prompt_data(self, prompt_data_name):
        prompt_data_file = prompt_data_name + '.jsonl'
        data_path = os.path.join(self.data_dir, prompt_data_file)
        assert os.path.exists(data_path), f"Data file {data_path} not found."
        data_df = pd.read_json(data_path, lines=True)
        return data_df
    
    def _single_input_for_single_token_prompt(self, instance):
            return str(instance['prompt']) + str(instance['number'])
    
    # def _sequence_input_for_sequence_prompt(self, instance):
    #     input = instance['prompt'] + ", ".join(instance['sequence'])
    #     target = instance['sequence']
    #     return input, target
    
    def get_number_token_idx_multi_token(self, input_text,tokenizer):
        inputs = tokenizer(input_text, return_tensors='pt')
        input_ids = inputs['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(f"Original input text: {input_text}")
        numbers = [match.span() for match in re.finditer(r'\d+', input_text)]

        number_token_idx = []
        for start, end in numbers:
            digit_str = input_text[start:end]
            print(f"Number '{digit_str}'")
            current_token_indices = []
            token_accumulator = ""

            # 循环遍历每个token
            for i, token in enumerate(tokens):
                decoded_token = tokenizer.decode([input_ids[i]]).strip()
                token_accumulator += decoded_token
                # 每遍历一个token,累计一个token,
                if token_accumulator == digit_str:
                    current_token_indices.append(i)
                    number_token_idx.append(current_token_indices)
                    break
                elif digit_str.startswith(token_accumulator):
                    current_token_indices.append(i)
                else:
                    token_accumulator = ""
                    current_token_indices = []
        for num_span, token_idx in zip(numbers, number_token_idx):
            token_texts = [tokens[i] for i in token_idx]
            print(f"Number '{input_text[num_span[0]:num_span[1]]}' is represented as'{token_texts}', is  at index {token_idx}")
        return number_token_idx

    def run_pipeline(self):
        # 1.判断是否存在已有的embedding文件
        embedding_result_fpath = os.path.join(self.embedding_dir, self.model+"."+self.prompt_data_name+'.pt')
        # 2.如果有，直接读取
        if os.path.exists(embedding_result_fpath):
            print(f"Embedding file {embedding_result_fpath} already exists. loading pt...")
            self.embedding = torch.load(embedding_result_fpath)
            return self.embedding
        
        # 3.如果没有，生成embedding文件, 初始化tokenizer和model
        # method = 'mean'
        method = self.method
        print(f'Start loading model: {self.model} and generating embedding: {self.prompt_data_name}.')
        tokenizer = AutoTokenizer.from_pretrained(model_fullnames[self.model],padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_fullnames[self.model])
        model.eval()
        if self.num_type_name == 'single':
            inputs = self.data_df.apply(self._single_input_for_single_token_prompt, axis=1)
            inputs_lst = inputs.tolist()        
            
            all_layer_embeddings = []
            for input_text in inputs_lst:
                layer_embedding = []
                # 获取数字token的index
                number_token_indices = self.get_number_token_idx_multi_token(input_text, tokenizer)
                print(f"Number token indices: {number_token_indices}")
                # 提取对应token的embedding
                encoded_inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**encoded_inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    for layer in hidden_states:
                        if method == 'last':
                            # number_token_indices = [[1,2,3,4]], extract the last token at"4"
                            embedding = layer[:, number_token_indices[-1][-1], :]
                            embedding = embedding.squeeze(dim=0) # remove batch dimension
                            layer_embedding.append(embedding)
                            print(f"embeddings shape: {np.array(embedding).shape}")
                        elif method == 'mean':
                            embedding = layer[0, number_token_indices, :].mean(dim=1)
                            embedding = embedding.squeeze(dim=0)
                            print(f"embeddings shape: {np.array(embedding).shape}")
                            layer_embedding.append(embedding)
                print(f"Layer embeddings shape: {np.array(layer_embedding).shape}") #should be (num_layers, embedding_dim)
                all_layer_embeddings.append(layer_embedding)
            all_layer_embeddings = np.transpose(all_layer_embeddings, (1, 0, 2))
            torch.save(all_layer_embeddings, embedding_result_fpath)
            print(f"Embeddings saved. Shapes:{np.array(all_layer_embeddings).shape}")
            self.embedding = all_layer_embeddings
            return self.embedding
        else:
            raise Exception("Invalid num_type_name.")
        
    def evaluate(self):
        results_fpath = os.path.join(self.results_dir, self.model+"."+self.prompt_data_name+'.jsonl')
        if os.path.exists(results_fpath):
            print(f"Results file {results_fpath} already exists.")
            # read redults file
            with open(results_fpath, 'r') as f:
                for line in f:
                    text = json.loads(line)
                    print(f"Layer:{text['embedding_layer']}, orderness:{text['orderness']}, spacing:{text['spacing']}, left_digit:{text['left_digit']}")
        
        embedding = self.embedding
        if self.num_type_name == 'single':
            labels = self.data_df['label'].tolist()
        elif self.num_type_name == 'sequence':
            labels = [i for i in range(0, 201)]

        layers = len(embedding)
        print(f"Number of layers: {layers}")
        results = []
        for i in range(layers):
            print(f"Layer {i}")
            embs = embedding[i]
            orderness = nearest_neighbor_analysis(embs, labels)
            spacing = distance_linear_analysis(embs, labels)
            left_digit = metric_left_digit_effect(embs, labels)
            result = {
                        'model': self.model,
                        'method': self.method,
                        'prompt_data_name': self.prompt_data_name,
                        'embedding_layer': i,    
                        'orderness': orderness, 
                        'left_digit': left_digit,
                        'spacing': spacing, 
                    }
            if self.is_probing:
                probe_compare = probe_num_compare_accu(embs, labels)
                exp_info = {
                    'model': self.model,
                    't_method': self.method,
                    'prompt': self.prompt_data_name,
                    'layer': i,
                    'orderness': orderness,
                    'spacing': spacing[1]/spacing[2],
                    'left_digit': math.log10(left_digit),
                } 
                probe_addition = probe_num_addition_accu(embs, labels, exp_info=exp_info)
                result['probe_compare'] = '%.4f' % probe_compare
                result['probe_addition'] = '%.4f' % probe_addition
            print(result)
            results.append(result)
        with open(results_fpath, 'w') as f:
            for result in results:
                f.write(json.dumps(result)+'\n')
