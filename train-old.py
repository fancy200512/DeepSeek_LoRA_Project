import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# 拉取预训练模型的地址，也可以是本地下载好的绝对路径
# 我这里HuggingFace底层服务限制了我的访问，可能是python自动下载较大文件会导致连接不稳定或者拒绝请求，因此我选择使用本地路径
MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B" 
DATASET_PATH = "/root/autodl-tmp/DeepSeek_LoRA_Project/data/huanhuan.json" # 数据集路径
OUTPUT_DIR = "./output/deepseek-lora-out" # 微调后的模型路径

print("加载模型与分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# Qwen/DeepSeek 系列通常没有默认的 pad_token，我们指定 eos_token 作为 pad_token
tokenizer.pad_token = tokenizer.eos_token

# 加载基座模型 (使用 bfloat16 以节省显存并加速)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, 
    device_map="auto",  # 自动分配到 GPU
    trust_remote_code=True
)

print("注入 LoRA 适配器...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               
    lora_alpha=32,      # 缩放系数，通常为 rank 的 2 倍
    lora_dropout=0.05,
    # 针对 Qwen 架构，通常将 LoRA 注入到注意力层和前馈层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
)
# 将基座模型包装成 PEFT 模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("加载并格式化数据集...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_instruction(example):
    """
    将 Alpaca 格式 (instruction, input, output) 转换为大模型能懂的对话文本。
    这里使用极其经典的 ChatML 格式。
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # 拼接用户提问
    if input_text:
        user_prompt = f"{instruction}\n{input_text}"
    else:
        user_prompt = instruction
        
    # 构建对话字符串
    text = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    return {"text": text}

# 映射数据集
formatted_dataset = dataset.map(format_instruction)

print("初始化训练器...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,      
    gradient_accumulation_steps=4,      
    learning_rate=5e-5,                 
    num_train_epochs=3,                 
    logging_steps=10,                   # 每 10 步打印一次 loss
    save_strategy="epoch",              # 每个 epoch 保存一次权重
    bf16=True,                          
    optim="adamw_torch",
    report_to="none"                    # 暂时关闭 wandb 监控
)

# 使用 trl 库的 SFTTrainer 进行监督微调
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    dataset_text_field="text",          # 指定数据集中包含文本的列名
    max_seq_length=512,                 # 限制最大序列长度防显存溢出
    tokenizer=tokenizer,
    args=training_args,
)

print("🚀 开始微调训练！")
trainer.train()

# 保存最终的 LoRA 权重
print("保存最终模型...")
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("✅ 训练完成！")
