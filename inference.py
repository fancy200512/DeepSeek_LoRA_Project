import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B"
LORA_PATH = "./output/deepseek-lora-out/final"

print("1. 加载基座模型与分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

print("2. 加载并融合 LoRA 权重...")
# 这一步对应了你 PPT 第 6 页的“模型合并”概念，但在推理时我们通常动态合并
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# 测试问题
question = "这个你不必着急，朕迟早会给你一个答复。"
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

print(f"\n用户: {question}")
print("甄嬛(思考中...):")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100,
        # 原始值为0.7
        temperature=0.7,
        repetition_penalty=1.1
    )

# 解码输出文本 (截取 assistant 后面的部分)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = response.split("assistant\n")[-1]
print(f"{answer}\n")