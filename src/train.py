import pandas as pd
from dotenv import load_dotenv

import torch 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer, TrainingArguments,
    BitsAndBytesConfig
)
from peft import ( 
    LoraConfig,
    get_peft_model
)

from datasets import Dataset
from trl import SFTTrainer, SFTConfig, setup_chat_format
from accelerate import Accelerator

load_dotenv()
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

model_checkpoint = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
new_model = model_checkpoint + '-ft'
train_data_path = 'data/train.csv'

def data_preprocessing(data_path, tokenizer, max_length=1024):
    def remove_newline(text):
        return text.replace('\n', ' ').replace('\r', ' ').strip()

    def chat_template(row):
        message = [
            {'role': 'user', 'content': row['Context']},
            {'role': 'assistant', 'content': row['Response']}
        ]
        return tokenizer.apply_chat_template(message, tokenize=False)

    data = pd.read_csv(data_path)
    data = data.dropna()
    data['Context'] = data['Context'].apply(remove_newline)
    data['Response'] = data['Response'].apply(remove_newline)
    data['Text'] = data.apply(chat_template, axis=1)
    dataset = Dataset.from_pandas(data)

    return dataset.train_test_split(test_size=0.1)

def main():
    device_index = Accelerator().process_index
    device_map = {'': device_index}
    torch_dtype = torch.float16
    attn_implementation = 'eager'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                                 quantization_config=bnb_config,
                                                 device_map=device_map,
                                                 attn_implementation=attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model, tokenizer = setup_chat_format(model, tokenizer)
    dataset = data_preprocessing(train_data_path, tokenizer)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )
    model = get_peft_model(model, peft_config)
    
    sft_config = SFTConfig(
        output_dir= f'log/{new_model}_output',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy='steps',
        eval_steps=0.2,
        logging_strategy='steps',
        logging_steps=0.5,
        warmup_steps=10,
        learning_rate=1e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to='none',
        
        dataset_batch_size=1,
        dataset_text_field='Text',
        max_seq_length=512,
        packing=False
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        peft_config=peft_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    trainer.train()
    model.save_pretrained(f'adapter/adapter-{model_checkpoint}/')


if __name__ == '__main__':
    main()