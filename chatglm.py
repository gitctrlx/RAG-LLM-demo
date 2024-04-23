from transformers import AutoTokenizer, AutoModel
from typing import Any
from llama_index.legacy.llms import CustomLLM, CompletionResponse, LLMMetadata, CompletionResponseGen
from llama_index.core.llms.callbacks import llm_completion_callback
from modelscope import snapshot_download

model_path = snapshot_download('ZhipuAI/chatglm3-6b', cache_dir="/root/autodl-tmp/")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model.eval()  # Ensuring the model is in evaluation mode

context_window = 2048
num_output = 256

class ChatGLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=context_window,
            num_output=num_output,
            model_name="chatglm3-6b",
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt_length = len(prompt)

        # only return newly generated tokens
        text,_ = model.chat(tokenizer, prompt, history=[])
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError()

# Initialize and test the custom LLM
if __name__ == "__main__":
    chat_glm = ChatGLM()
    test_prompt = "Hello, how are you?"
    response = chat_glm.complete(test_prompt)
    print("Generated text:", response.text)
