import os
# 消除终端里的 libgomp 警告
os.environ["OMP_NUM_THREADS"] = "1" 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B"
LORA_PATH = "./output/het_rank_focal_out/final"

print("1. 加载基座模型与分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("2. 加载异构秩 LoRA 权重...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

question = "多谢姐姐出言相助。今日之恩，没齿难忘。"
# 强制引导模型进入思维链模式
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"

print(f"\n用户: {question}")
print("甄嬛 (内心盘算中，请稍候...)")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 修正 1：获取正确的结束符 ID
# 优先使用模型默认的 eos_token，同时兼容将 <|im_end|> 转换为标准 ID
eos_ids = [tokenizer.eos_token_id]
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
if im_end_id is not None:
    eos_ids.append(im_end_id)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256, 
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=eos_ids, # 使用修正后的合法结束符列表
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

# 仅解码新生成的部分
generated_ids = outputs[0][inputs.input_ids.shape[1]:]

# 修正 2：必须设为 False！保留 <think> 等标签，供我们后续切分
full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

# （可选）你可以取消下面这行的注释，看看模型最原始的生成文本是什么样的
# print("\n[Debug 原始输出]：", full_text)

# 修正 3：干净地解析出最终回答，并顺手除掉可能残存的 <|im_end|>
if "<answer>" in full_text and "</answer>" in full_text:
    answer = full_text.split("<answer>")[-1].split("</answer>")[0]
elif "</think>" in full_text:
    answer = full_text.split("</think>")[-1].replace("<answer>", "")
else:
    answer = full_text

# 清理两端空格和潜在的结束符
answer = answer.replace("<|im_end|>", "").strip()

print("\n--- 最终回复 ---")
print(f"甄嬛: {answer}\n")