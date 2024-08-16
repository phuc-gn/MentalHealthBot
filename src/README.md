For distributed training, I use the `accelerate` library. To run the training script, use the following command:

```bash
accelerate launch train.py
```

The adapter will be saved in the `adapter` directory. I also uploaded the adapter to the Hugging Face model hub, so you can use it directly in your model. The adapter name is [pgnguyen/llama-3.1-8b-mental](https://huggingface.co/pgnguyen/llama-3.1-8b-mental).