from llama_cpp import Llama

class LlamaModel:
    def __init__(self, model_path, n_gpu_layers):
        """
        Initialize and load the Llama model.

        Args:
            model_path (str): Path to the model file.
        """
        self.llm = Llama(model_path=model_path,n_gpu_layers=n_gpu_layers)

    def generate(self, prompt, max_tokens):
        """
        Generate text using the loaded Llama model.

        Args:
            prompt (str): The input prompt for text generation.
            max_tokens (int): Maximum number of tokens to generate.

        Returns:
            dict: The generated output from the model.
        """
        return self.llm(prompt, max_tokens=max_tokens)
