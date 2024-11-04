import sys
# 将 GPTQ-for-LLaMa-triton 文件夹路径添加到 Python 环境的第一个位置
sys.path.insert(0, '/zzp/GPTQ-for-LLaMa-triton')  # 替换为实际路径
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from quant.quant_linear import QuantLinear
from utils.modelutils import torch_snr_error
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 设置模型路径
model_name_or_path = "/zzp/smh/pyworks/llama13"  # 替换为 LLaMA 模型的实际路径

# 直接在 GPU 上加载模型并应用 4 位量化
print("开始加载模型到 GPU，并应用 4 位量化...")
# 尝试直接使用 4 位量化配置
quantization_config = BitsAndBytesConfig(load_in_4bit=True)  # 使用 4 位量化
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quantization_config,
    device_map={"": "cuda"}
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print("模型加载完成。")
# 初始化 GPTQ 量化器
class GPTQForLLaMA:
    def __init__(self, model, bits=4, groupsize=-1):  # 新增 groupsize 参数
        self.model = model
        self.bits = bits
        self.groupsize = groupsize

    def quantize_model(self):
        # 逐层量化并释放未使用的显存
        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Linear):  # 仅量化 Linear 层
                in_features = layer.in_features
                out_features = layer.out_features
                bias = layer.bias is not None

                # 创建量化层
                quant_layer = QuantLinear(self.bits, self.groupsize, in_features, out_features, bias)
                
                # 将权重移到 GPU 并赋值到量化层
                quant_layer.qweight = layer.weight.clone().to("cuda")
                if bias:
                    quant_layer.bias = layer.bias.clone().to("cuda")

                # 替换原始层为量化层
                parent_module, child_name = self._get_parent_module(name)
                setattr(parent_module, child_name, quant_layer)

                # 将原始层权重移回 CPU 并释放未使用的显存
                layer.to("cpu")
                torch.cuda.empty_cache()

    def _get_parent_module(self, name):
        """辅助函数，用于获取父模块及其子模块的名称，以便替换层"""
        name_parts = name.split(".")
        module = self.model
        for part in name_parts[:-1]:  # 遍历到倒数第二个部分
            module = getattr(module, part)
        return module, name_parts[-1]


# 量化模型
quantizer = GPTQForLLaMA(model, bits=4)
with torch.no_grad():  # 避免计算梯度，节省显存
    quantizer.quantize_model()
print("模型量化完成。")

print(f"量化完成后显存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")




