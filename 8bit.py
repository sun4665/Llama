import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 模型路径
model_name_or_path = "/zzp/smh/pyworks/llama13"

# 打印加载前的 GPU 内存使用情况
print(f"加载8位量化模型前，显存使用: {torch.cuda.memory_allocated() / 1024**2} MB")
extra_tensors = [torch.rand((12247, 12247), dtype=torch.float32, device="cuda") for _ in range(10)]
# 使用 8 位量化配置加载模型
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 加载 8 位量化模型
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quantization_config,  # 使用 8 位量化
    device_map="auto",
)

# 打印加载后的 GPU 内存使用情况
print(f"加载8位量化模型后，显存使用: {torch.cuda.memory_allocated() / 1024**2} MB")

# 加载与模型匹配的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 示例推理文本
input_text = "Hello, how are you?"

# 编码输入文本为模型可接受的格式
inputs = tokenizer(input_text, return_tensors="pt")

# 将 inputs 移动到模型所在的设备上 (GPU)
inputs = inputs.to(model.device)

# 使用量化后的模型进行推理
outputs = model.generate(**inputs, max_length=50)

# 解码并输出生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
