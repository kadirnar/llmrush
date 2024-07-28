import os
import time
import warnings

import GPUtil
import torch
from colorama import Fore, Style, init
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize colorama for cross-platform colored terminal output
init()


class LLMInference:

    def __init__(
        self,
        model_name: str,
        gpu_ids: list = [0],
        use_4bit: bool = True,
        max_new_tokens: int = 100,
    ):
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens

        self.load_model()

    def load_model(self):
        if torch.cuda.is_available() and self.gpu_ids:
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
        else:
            self.device = torch.device("cpu")
            print(f"{Fore.YELLOW}Using CPU for inference.{Style.RESET_ALL}")

        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.use_4bit, bnb_4bit_compute_dtype=torch.float16)

        # Suppress stdout temporarily
        with open(os.devnull, "w") as devnull:
            old_stdout = os.dup(1)
            os.dup2(devnull.fileno(), 1)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            finally:
                os.dup2(old_stdout, 1)

        if len(self.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)

        # Set pad_token_id if it's not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.log_gpu_memory()

    def log_gpu_memory(self):
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            for gpu_id in self.gpu_ids:
                gpu = gpus[gpu_id]
                print(
                    f"{Fore.MAGENTA}GPU {gpu_id} VRAM: {gpu.memoryUsed:.2f} MB / {gpu.memoryTotal:.2f} MB{Style.RESET_ALL}"
                )

    def generate_response(self,
                          system_prompt: str,
                          user_prompt: str,
                          do_sample: bool = True) -> tuple[str, float]:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ]

        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = inputs.to(self.device)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0

        input_length = input_ids.shape[1]

        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=do_sample,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        end_time = time.time()

        response = self.tokenizer.decode(generated_ids[0, input_length:], skip_special_tokens=True)

        num_tokens = len(generated_ids[0]) - input_length
        time_taken = end_time - start_time
        tokens_per_second = num_tokens / time_taken

        return response, tokens_per_second


# Usage example
if __name__ == "__main__":
    model_name = "llama-3-8b-Instruct-bnb-4bit"
    gpu_ids = [0]  # Use GPUs 2 and 3. Modify as needed.
    llm = LLMInference(model_name=model_name, gpu_ids=gpu_ids, use_4bit=True)

    while True:
        system_prompt = input(f"\n{Fore.BLUE}Enter system prompt (or 'quit' to exit): {Style.RESET_ALL}")
        if system_prompt.lower() == "quit":
            break

        user_prompt = input(f"{Fore.GREEN}Enter user prompt: {Style.RESET_ALL}")

        response, tokens_per_second = llm.generate_response(system_prompt, user_prompt)

        print(f"\n{Fore.BLUE}System:{Style.RESET_ALL} {system_prompt}")
        print(f"{Fore.GREEN}User:{Style.RESET_ALL} {user_prompt}")
        print(f"{Fore.YELLOW}Assistant:{Style.RESET_ALL} {response}")
        print(f"{Fore.CYAN}Tokens per second:{Style.RESET_ALL} {tokens_per_second:.2f}")
