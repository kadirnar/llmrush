from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformerTextGenerator:
    """
    A class to handle text generation using the Transformers library with LoRA support.

    This class provides functionality to load a specified model with an optional LoRA adapter and generate
    text responses.
    """

    def __init__(
            self,
            model_id: str = "meta-llama/Meta-Llama-3.1-8B",
            load_in_4bit: bool = False,
            low_cpu_mem: bool = False,
            lora_path: Optional[str] = None):
        """
        Initialize the TransformerTextGenerator with a specified model and optional LoRA adapter.

        Args:
            model_id (str): The identifier of the model to use.
            load_in_4bit (bool): Whether to load the model in 4-bit precision.
            low_cpu_mem (bool): Whether to use low CPU memory usage.
            lora_path (Optional[str]): Path to a pre-trained LoRA adapter.
        """
        self.model_id: str = model_id
        self.model = None
        self.tokenizer = None
        self.load_in_4bit = load_in_4bit
        self.low_cpu_mem = low_cpu_mem
        self.lora_path = lora_path

        if self.model is None:
            self._load_model()

    def _load_model(self) -> None:
        """
        Load the language model, tokenizer, and LoRA adapter if specified.

        This method initializes the model with bfloat16 precision and loads the LoRA adapter if a path is
        provided.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model_kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": self.low_cpu_mem}

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

        if self.lora_path:
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)

    def __call__(
            self,
            system_prompt: str,
            user_prompt: str,
            max_new_tokens: int = 256,
            temperature: float = 0.6,
            top_p: float = 0.9) -> str:
        """
        Generate a text response based on the system and user prompts.

        Args:
            system_prompt (str): The system prompt to set the context.
            user_prompt (str): The user's input prompt.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 256.
            temperature (float, optional): Controls randomness in generation. Higher values make output more random. Defaults to 0.6.
            top_p (float, optional): Controls diversity of generated tokens. Defaults to 0.9.

        Returns:
            str: The generated text response.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded. There was an error during initialization.")

        prompt = f"System: {system_prompt}\nHuman: {user_prompt}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output
