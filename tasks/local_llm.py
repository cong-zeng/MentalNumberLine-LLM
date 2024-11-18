

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

model_fullnames = {  'gpt2': 'openai-community/gpt2',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'opt-13b': 'facebook/opt-13b',
                     'math_gpt2': 'Sharathhebbar24/math_gpt2',
                     'mathstral-7B': 'mistralai/Mathstral-7B-v0.1',
                     'opt-1.3b':"facebook/opt-iml-1.3b",
                     'llama3-8b':'meta-llama/Meta-Llama-3-8B-Instruct',
                     'mistral-7B':"mistralai/Mistral-7B-Instruct-v0.3",
                     'llama3.1-8b':'meta-llama/Llama-3.1-8B-Instruct',
                     'Phi3.5-4b':"microsoft/Phi-3.5-mini-instruct",
                     'Qwen2.5-7B': "Qwen/Qwen2.5-7B-Instruct",
                     'gemma2-9b':"google/gemma-2-9b-it",
                     }

def multi_call_local_model(model_name, inputs):
    tokenizer = AutoTokenizer.from_pretrained(model_fullnames[model_name],padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_fullnames[model_name], torch_dtype=torch.float16)
    print(f"Model: {model_name} loaded successfully")

    # Move model and inputs to GPU if available
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda:1")
        model = model.to(device)

    # Tokenize inputs in smaller batches to avoid memory spikes
    batch_size = 8  # Adjust this based on your memory constraints
    outputs = []

    for i in tqdm(range(0, len(inputs), batch_size), desc="Generating outputs"):
        batch_inputs = inputs[i:i+batch_size]
        
        # Tokenize the batch
        tokenized_inputs = tokenizer(batch_inputs.tolist(), return_tensors='pt', padding=True, truncation=True)
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
        
        # Generate outputs with no_grad to save memory
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=tokenized_inputs['input_ids'],
                attention_mask=tokenized_inputs['attention_mask'],
                max_length=256,  # Adjust max length as needed
                num_return_sequences=1
            )
        
        # Decode outputs
        decoded_outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        
        # Remove input text from outputs if they start with it
        for j, input_text in enumerate(batch_inputs):
            if decoded_outputs[j].startswith(input_text):
                decoded_outputs[j] = decoded_outputs[j][len(input_text):]
        
        # Collect outputs
        outputs.extend(decoded_outputs)
    del model
    torch.cuda.empty_cache()
    return outputs