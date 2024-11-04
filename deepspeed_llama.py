import torch
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, TrainerCallback, BitsAndBytesConfig
from datasets import Dataset
import os
import deepspeed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from peft import LoraConfig, get_peft_model  # 导入 PEFT 相关模块

logging.basicConfig(level=logging.DEBUG)
os.environ["DEEPSPEED_LOG_LEVEL"] = "info" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_DISABLE"] = "True"
print(torch.cuda.is_available())
torch.set_autocast_enabled(False)

# 加载LLaMA模型和分词器
model_name = "/zzp/smh/pyworks/llama"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 使用 8 位量化配置
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 直接在 GPU 上加载模型并应用 8 位量化
model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map={"": "cuda:0"} 
)

# 使用 PEFT 配置 LoRA 层
lora_config = LoraConfig(
    r=16,  # 可以调整的 LoRA 矩阵的秩
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 要添加适配器的层，取决于模型结构
    lora_dropout=0.05,
    bias="none"
)

# 将模型转换为带有 LoRA 适配器的 PEFT 模型
model = get_peft_model(model, lora_config)

# 将 LlamaTokenizer 的 pad_token 设置为 eos_token
tokenizer.pad_token = tokenizer.eos_token

# 加载数据集
def load_dataset():
    df = pd.read_parquet("/zzp/smh/pyworks/wikitext__wikitext-2-raw-v1/data/train-00000-of-00001-6506f33274247c0c.parquet")
    dataset = []
    for _, row in df.iterrows():
        input_ids = tokenizer.encode(row['text'], truncation=True, padding="max_length", max_length=16)
        labels = input_ids.copy()  # 自回归模型，标签与输入相同
        dataset.append({"input_ids": input_ids, "labels": labels})
    return Dataset.from_pandas(pd.DataFrame(dataset))

dataset = load_dataset()

# 自定义一个Callback类来保存损失值
class LossCallback(TrainerCallback):
    def __init__(self):
        self.loss_values = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs:
            self.loss_values.append(logs["loss"])

# DeepSpeed 配置文件
deepspeed_config = {
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "fp16": {"enabled": "auto"},  # 将 fp16 设置为 'auto'
    "optimizer": {
        "type": "AdamW",  # 设置为 AdamW 优化器
        "params": {
            "lr": 8e-6,           # 学习率，与 TrainingArguments 中的学习率一致
            "betas": [0.9, 0.999], # 默认 beta 值
            "eps": 1e-8,           # epsilon 值
            "weight_decay": 0.01   # 权重衰减，与 TrainingArguments 中的值一致
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_bucket_size": 2e8,
        "reduce_bucket_size": 2e8,
    }
  
}


# 训练参数配置  
training_args = TrainingArguments(
    output_dir="./llama_deepspeed_output",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=8e-6,
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,  # 设置为 True 或 False，根据需要
    save_steps=1000,
    evaluation_strategy="no",
    report_to="none",
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    deepspeed=deepspeed_config  # 启用 DeepSpeed 并使用配置文件
)


# 初始化LossCallback
loss_callback = LossCallback()

# 使用DeepSpeed支持初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[loss_callback]
)

# 开始训练
trainer.train()

# 训练结束后绘制损失曲线
plt.plot(loss_callback.loss_values)
plt.title("Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")

# 保存图像为文件
plt.savefig("training_loss.png")
print("损失曲线已保存为 'training_loss.png'")











