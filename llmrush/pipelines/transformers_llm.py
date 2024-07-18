import torch
from huggingface_hub import snapshot_download
from transformers import pipeline


class PirateChatbot:

    def __init__(
            self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", token="hf_token", local_dir="llm_weights"):
        self.model_id = model_id
        self.token = token
        self.model_file = None
        self.pipe = None
        self.tokenizer = None

        self.download_model(local_dir=local_dir)

        if self.pipe is None:
            self.load_model()

    def download_model(self, local_dir="llm_weights"):
        self.model_file = snapshot_download(
            repo_id=self.model_id,
            repo_type="model",
            ignore_patterns=["*.md", "*.gitattributes"],
            local_dir=local_dir,
            token=self.token,
        )

    def load_model(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model_file,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {
                    "load_in_4bit": True
                },
            },
        )

    def generate_response(self, messages, max_new_tokens=256, temperature=0.6, top_p=0.9):
        terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        return outputs[0]["generated_text"][-1]["content"]

    def chat(self, user_input):
        messages = [
            {
                "role": "system",
                "content": "You are a pirate chatbot who always responds in pirate speak!"
            },
            {
                "role": "user",
                "content": user_input
            },
        ]
        return self.generate_response(messages)
