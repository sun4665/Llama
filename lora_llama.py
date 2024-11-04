import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import Adam8bit

# 检查 GPU 是否可用，并设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 设置 CUDA 环境变量，调整 max_split_size_mb 来避免内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 模型文件夹路径
model_path = "/zzp/smh/pyworks/llama13"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# 检查并设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 设置 8-bit 量化的配置
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 加载模型，并使用 8-bit 量化
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    local_files_only=True, 
    quantization_config=quantization_config  # 使用 BitsAndBytesConfig
)

# 启用梯度检查点，减少显存占用
model.gradient_checkpointing_enable()

# 设置模型参数是否需要梯度更新
for param in model.parameters():
    if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        param.requires_grad = True

# 定义数据集目录路径
data_dir = "/zzp/smh/pyworks/wikitext__wikitext-2-raw-v1/data"

# 加载数据集文件
train_df = pd.read_parquet(os.path.join(data_dir, "train-00000-of-00001-6506f33274247c0c.parquet"))
validation_df = pd.read_parquet(os.path.join(data_dir, "validation-00000-of-00001-ee9a0a71dcb41f66.parquet"))

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)

# 数据预处理函数，添加 labels
def preprocess_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=16)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# 对数据集进行分词处理
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = validation_dataset.map(preprocess_function, batched=True)

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,  
    target_modules=["q_proj", "v_proj"]
)

# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)
# 确保模型参数启用了梯度计算，但只对浮点数类型张量设置 requires_grad=True
for param in model.parameters():
    if param.dtype.is_floating_point or param.dtype.is_complex:
        param.requires_grad = True

# 打印模型参数所在的设备
for name, param in model.named_parameters():
    print(f"Parameter {name} is on device: {param.device}")

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",  
    eval_steps=20,
    save_total_limit=3,  
    learning_rate=8e-6,
    per_device_train_batch_size=16,  
    num_train_epochs=5,
    max_steps=800, #迭代次数 
    gradient_accumulation_steps=16,  
    logging_dir="./logs",
    logging_steps=20,
    lr_scheduler_type="cosine",  # 使用余弦学习率调度器
    warmup_steps=500,  # 添加热身步数，训练初期缓慢升高学习率
    fp16=False,  
    remove_unused_columns=False,
    dataloader_pin_memory=False,  
    no_cuda=False, 
    weight_decay=0.01,  # 添加正则化 
)

# 提前终止回调
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
# 使用8-bit优化器
optimizer = Adam8bit(model.parameters(), lr=8e-6)
# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, None)  # 使用8-bit优化器
)

# 开始训练
train_result = trainer.train()
print("Log history:")
for log in trainer.state.log_history:
    print(log)

# 绘制并保存训练损失图像
def plot_loss(logs):
    train_loss = []
    eval_loss = []
    steps = []
    epochs = []

    for log in logs:
        if 'loss' in log:
            train_loss.append(log['loss'])
        if 'eval_loss' in log:
            eval_loss.append(log['eval_loss'])
        if 'step' in log:
            steps.append(log['step'])
        if 'epoch' in log:
            epochs.append(log['epoch'])

    # 打印收集到的数据进行检查
    print(f"Train loss: {train_loss}")
    print(f"Steps: {steps}")
    print(f"Eval loss: {eval_loss}")
    print(f"Epochs: {epochs}")

    # 确保 steps 和 train_loss 的长度一致
    min_length = min(len(steps), len(train_loss))
    steps = steps[:min_length]
    train_loss = train_loss[:min_length]

    plt.plot(steps, train_loss, label="train_loss")

    # 确保 eval_loss 和 steps 的长度一致
    if eval_loss:
        min_eval_length = min(len(steps), len(eval_loss))
        eval_loss = eval_loss[:min_eval_length]
        plt.plot(steps[:min_eval_length], eval_loss, label="eval_loss", linestyle="--")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Evaluation Loss")

    # 保存图像为文件
    plt.savefig("loss_plot.png")
    plt.show()

# 调用绘图函数
plot_loss(trainer.state.log_history)
