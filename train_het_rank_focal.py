import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ================= 1. 参数 =================
MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B"
DATASET_PATH = "./data/huanhuan_cot.json"
OUTPUT_DIR = "./output/het_rank_focal_out"

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
# 开启梯度检查点，用计算时间换取显存空间，防止 AutoDL 上 OOM
model.gradient_checkpointing_enable()

# ================= 3. 标准化异构秩注入 =================
print("注入异构 LoRA 适配器（Attention: r=8, MLP: r=32）")
# 官方推荐写法：在一个 Config 中通过字典精确分配，避免多 Adapter 冲突
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,              # 默认基础秩
    lora_alpha=16, 
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    rank_pattern={
        "gate_proj": 32, "up_proj": 32, "down_proj": 32
    },
    alpha_pattern={
        "gate_proj": 64, "up_proj": 64, "down_proj": 64
    }
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= 4. 数据集处理 =================
print("处理数据集...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_instruction(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example["output"]   
    user_prompt = f"{instruction}\n{input_text}" if input_text else instruction
    text = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    return {"text": text}

formatted_dataset = dataset.map(format_instruction)

# ================= 5. 自定义 Focal Loss Trainer =================
class FocalSFTTrainer(SFTTrainer):
    def __init__(self, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits                       
        labels = inputs.get("labels")                 

        if labels is None:
            return outputs.loss                      

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100
        )

        # Token-Level Focal Loss 核心计算
        probs = torch.exp(-ce_loss)                  
        pt = torch.clamp(probs, min=1e-9, max=1.0)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        valid_mask = (shift_labels.view(-1) != -100)
        if valid_mask.sum() > 0:
            loss = focal_loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return (loss, outputs) if return_outputs else loss

# ================= 6. 训练参数设置 =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="adamw_torch",
    report_to="none",
    gradient_checkpointing=True # 再次确保在 args 中开启
)

trainer = FocalSFTTrainer(
    gamma=2.0,                     
    model=model,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=768,
    tokenizer=tokenizer,
    args=training_args,
)

# ================= 7. 开始训练 =================
print("🚀 开始异构秩 + Focal Loss 训练！")
trainer.train()

print("💾 保存最终模型权重...")
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("✅ 训练完成！")