import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback # 新增：导入早停回调函数
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ================= 1. 参数配置 =================
MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B" 
DATASET_PATH = "/root/autodl-tmp/DeepSeek_LoRA_Project/data/huanhuan.json"
OUTPUT_DIR = "./output/deepseek-lora-train-out"

# ================= 2. 加载模型与分词器 =================
print("加载模型与分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

# ================= 3. 配置 LoRA 算法 =================
print("注入 LoRA 适配器...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= 4. 数据集处理与切分 =================
print("加载并格式化数据集...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# 新增：从 18000 多条数据中切分出 5% (约 900 条) 作为验证集，用于早停监控
split_dataset = dataset.train_test_split(test_size=0.05, seed=42)

def format_instruction(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        user_prompt = f"{instruction}\n{input_text}"
    else:
        user_prompt = instruction
        
    text = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    return {"text": text}

# 分别映射训练集和验证集
train_dataset = split_dataset["train"].map(format_instruction)
eval_dataset = split_dataset["test"].map(format_instruction)

# ================= 5. 设置训练参数 =================
print("初始化训练器...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    
    # 修改：针对中等数据集与 LoRA，学习率调整为 1e-4，Epoch 上限设为 8
    learning_rate=1e-4,                 
    num_train_epochs=8,                 
    
    # 新增：验证与保存策略配置 (早停必备)
    evaluation_strategy="epoch",        # 每个 epoch 跑完后在验证集上评估一次 loss
    save_strategy="epoch",              # 保存策略必须与评估策略保持一致
    load_best_model_at_end=True,        # 训练因早停结束后，自动加载验证集 loss 最小的那一次权重
    metric_for_best_model="loss",       # 评判最优模型的指标为 loss
    
    logging_steps=10,
    bf16=True,
    optim="adamw_torch",
    
    # 修改：开启 TensorBoard 以便后续观察双 loss 曲线
    report_to="tensorboard"             
)

# 使用 trl 库的 SFTTrainer 进行监督微调
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,        # 传入训练集
    eval_dataset=eval_dataset,          # 新增：传入验证集
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    # 新增：设置早停机制，patience=3 (连续 3 个 epoch 验证集 loss 不降则刹车)
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
)

# ================= 6. 开始训练 =================
print("🚀 开始微调训练！")
trainer.train()

# 保存最终的最优 LoRA 权重
print("保存最终最优模型...")
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("✅ 训练完成！")