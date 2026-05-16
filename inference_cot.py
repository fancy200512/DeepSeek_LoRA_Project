import os
os.environ["OMP_NUM_THREADS"] = "1" 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B"
LORA_PATH = "./output/het_rank_focal_out/final"

print("加载基座模型与分词器...")
# 加载 BPE 分词器，允许执行远程仓库的自定义代码流
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# 实例化基座大语言模型
# 使用 bfloat16 半精度浮点数加载，大幅降低推理显存占用的同时保证数值稳定性
# 自动接管底层设备的张量分配
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("加载异构秩 LoRA 权重...")
# 采用旁路融合机制，将训练好的异构秩 LoRA 矩阵动态挂载到冻结的基座模型上
model = PeftModel.from_pretrained(base_model, LORA_PATH)
# 强制模型进入评估模式，冻结 Dropout 等随机层，确保推理的确定性
model.eval()

question = "多谢姐姐出言相助。今日之恩，没齿难忘。"

# 在 prompt 尾部强制写入 <think> 标签，利用自回归模型的特性，诱导其优先输出思维链逻辑
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"

print(f"\n用户: {question}")
print("甄嬛 (内心盘算中，请稍候...)")

# 将自然语言提示词转换为高维张量，并推送到 GPU 显存中
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


eos_ids = [tokenizer.eos_token_id]
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
if im_end_id is not None:
    eos_ids.append(im_end_id)

# 开启无梯度上下文，彻底释放反向传播所需的计算图显存
with torch.no_grad():
    # 触发模型底层的自回归生成管线
    outputs = model.generate(
        # 预留足够的 token 长度以容纳“内部独白”与“最终回复”两段式输出
        max_new_tokens=256, 
        # 设置采样温度，0.7 可在维持角色逻辑严密性的同时赋予语言一定的创造力
        temperature=0.7,
        # 引入重复惩罚机制，抑制复读机现象
        repetition_penalty=1.1,
        # 传入修正后的结束符池
        eos_token_id=eos_ids, 
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

# 利用张量切片操作，剥离输入的 prompt 提示词，精准截取模型本次全新生成的 Token ID
generated_ids = outputs[0][inputs.input_ids.shape[1]:]

# 张量解码还原文本
# skip_special_tokens，以保留底层的 <think> 和 <answer> 结构化标签供业务层正则解析
full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

# 结构化文本解析路由
# 优先精准匹配闭合的 <answer> 标签；若模型偶尔漏掉，则降级通过 </think> 边界进行字符串分割提取
if "<answer>" in full_text and "</answer>" in full_text:
    answer = full_text.split("<answer>")[-1].split("</answer>")[0]
elif "</think>" in full_text:
    answer = full_text.split("</think>")[-1].replace("<answer>", "")
else:
    answer = full_text

# 抹除解码过程中可能残存的系统级对话停止符，并清理两侧空白字符
answer = answer.replace("<|im_end|>", "").strip()

print("\n--- 最终回复 ---")
print(f"甄嬛: {answer}\n")
