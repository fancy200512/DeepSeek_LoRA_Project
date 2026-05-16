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

MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B"
DATASET_PATH = "./data/huanhuan_cot.json"
OUTPUT_DIR = "./output/het_rank_focal_out"

print("加载模型与分词器...")
# 加载分词器并统一填充符与结束符
# 避免在定长截断或变长序列批处理（Batching）时引发底层张量维度对齐错误
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 实例化基座大语言模型
# 采用 bfloat16 混合精度加载，有效防范大模型训练中极易发生的梯度溢出（NaN）问题
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 开启梯度检查点机制
# 底层原理是用前向传播的额外计算时间来换取反向传播时的显存空间，是受限算力下单卡微调的策略
model.gradient_checkpointing_enable()

print("注入异构 LoRA 适配器（Attention: r=8, MLP: r=32）")
# rank_pattern 与 alpha_pattern 实现非对称矩阵旁路注入
# 对承载基础语义理解的 Attention 层分配低秩 r=8，保持轻量
# 对主导复杂逻辑推理与思维链（CoT）生成的 MLP 层分配高秩 r=32，显著拓宽其非线性表征空间
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
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

# 将异构 LoRA 结构挂载至冻结的基座模型之上
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("处理数据集...")
# 加载通过知识蒸馏合成的双段式 CoT 本地数据集
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# 定义指令格式化函数
# 将 JSON 字段严格拼接为 ChatML 多轮对话模板，为模型提供清晰的角色边界与思维链路指示
def format_instruction(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example["output"]   
    user_prompt = f"{instruction}\n{input_text}" if input_text else instruction
    text = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    return {"text": text}

# 利用 map 算子进行高并发的结构化数据映射
formatted_dataset = dataset.map(format_instruction)

# 深度重构底层训练算子，引入词元级焦点损失（Token-Level Focal Loss）机制
class FocalSFTTrainer(SFTTrainer):
    def __init__(self, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # gamma 调节因子：值越大，对高置信度（如“的”、“了”等高频虚词）的梯度压制就越强烈
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 获取模型的前向传播高维张量输出及真实标签
        outputs = model(**inputs)
        logits = outputs.logits                       
        labels = inputs.get("labels")                 

        if labels is None:
            return outputs.loss                      

        # 错位对齐计算自回归损失
        # 抛弃 logits 最后一个预测结果与 labels 第一个输入，使得前一个词刚好预测后一个词
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 计算基础的交叉熵损失
        # ignore_index=-100指示模型在计算 Loss 时主动屏蔽掉提问部分，只对生成的回复部分计算梯度
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100
        )

        # Token-Level Focal Loss 核心计算
        # 将交叉熵逆向推导，还原为模型对当前词预测的概率 pt
        probs = torch.exp(-ce_loss)                  
        # 引入截断保护机制，防止极小值导致的底层计算下溢或触发 NaN 崩溃
        pt = torch.clamp(probs, min=1e-9, max=1.0)
        # 计算焦点权重：模型越确定的通用词（pt越接近1），其权重被压降得越狠
        focal_weight = (1 - pt) ** self.gamma
        # 将动态权重乘回基础损失，强制迫使反向传播梯度向低频长尾风格词元转移
        focal_loss = focal_weight * ce_loss

        # 利用掩码过滤掉 -100 的无效位置，仅对有效响应区域的损失求取均值
        valid_mask = (shift_labels.view(-1) != -100)
        if valid_mask.sum() > 0:
            loss = focal_loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # 单卡设备批处理大小，配合梯度累加可等效扩大全局 Batch Size
    per_device_train_batch_size=2,
    # 梯度累加步数：每 4 步才进行一次真实的权重更新，以时间换取更大的全局优化视野
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="adamw_torch",
    report_to="none",
    # 确保在运行参数中再次拉起梯度检查点机制
    gradient_checkpointing=True 
)

# 挂载定制化的损失函数计算管线
trainer = FocalSFTTrainer(
    gamma=2.0,                     
    model=model,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=768,
    tokenizer=tokenizer,
    args=training_args,
)

print("🚀 开始异构秩 + Focal Loss 训练！")
# 启动反向传播及梯度更新链路
trainer.train()

print("💾 保存最终模型权重...")
# 训练结束后，固化最新的 LoRA 参数和词表字典落盘
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("✅ 训练完成！")
