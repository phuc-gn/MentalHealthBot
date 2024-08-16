from flask import Flask, request, jsonify
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from trl import setup_chat_format

model_checkpoint = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
lora_checkpoint = 'pgnguyen/llama-3.1-8b-mental'

load_dotenv()
app = Flask(__name__)
base_model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            torch_dtype=torch.float16,
            device_map='auto'
            )
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

base_model, tokenizer = setup_chat_format(base_model, tokenizer)
model = PeftModel.from_pretrained(base_model, lora_checkpoint)
model = model.merge_and_unload()
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 150)
    
    messages = [{"role": "user", "content": prompt}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    assistant_response = outputs[0]["generated_text"].split("assistant\n")[1].strip()
    
    return jsonify({'text': assistant_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)