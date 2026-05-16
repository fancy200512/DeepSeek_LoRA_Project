import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B"  # 本地基座模型的绝对路径
LORA_PATH = "./output/deepseek-lora-out/final" # 微调训练结束后保存的 LoRA 权重文件夹路径

print("1. 加载基座模型与分词器...")

# 加载分词器 (Tokenizer)
# trust_remote_code=True: 允许加载并执行模型开发者自定义的分词逻辑代码
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# 加载大语言模型作为基座
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    # torch_dtype: 使用 bfloat16 精度加载模型，能大幅减少显存占用并保持与训练时相同的推理精度
    torch_dtype=torch.bfloat16, 
    # 自动识别当前机器的 GPU 硬件资源，并将模型权重合理分配到显存中
    device_map="auto",
    trust_remote_code=True
)

print("2. 加载并融合 LoRA 权重...")

# 加载微调后的 LoRA 适配器权重并与基座模型动态组合
# 这种加载方式不会破坏基座模型原有的底层参数，而是通过旁路矩阵相乘的方式将你微调学到的角色风格叠加到基座上
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# 将模型设置为评估模式，关闭 Dropout 等训练专用的随机机制，确保输出状态的稳定性
model.eval()

# 构建测试问题与对话模板
question = "这个你不必着急，朕迟早会给你一个答复。"

# 按照ChatGML指令模板拼接字符串
# 只有模板的标签格式与训练时完全一致，模型才能正确识别出自己现在需要扮演 assistant 角色并开始作答
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

print(f"\n用户: {question}")
print("甄嬛(思考中...):")

# 将拼装好的文本 prompt 转换为模型能理解的数字 ID 张量，并推入 GPU 显卡计算
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 开启无梯度模式，节约显存并加速整个推理过程
with torch.no_grad():
    # 调用模型底层的自回归生成函数开始逐字推理
    outputs = model.generate(
        # 将输入张量展开并传入
        # 限制模型本次回答最多生成的新词汇数量
        max_new_tokens=100,
        # 采样温度，控制输出的多样性和创造力。值越低回答越严谨死板，值越高越具随机性。0.7 适合角色扮演任务
        temperature=0.7,
        # 重复惩罚因子。设为 1.1 可以有效压制大模型在局部文本中像复读机一样反复输出同一个词的概率
        repetition_penalty=1.1
    )

# 解码输出文本并进行后处理
# 在解码时自动过滤掉类似 <|im_end|> 这种原本用于控制格式的特殊触发符，让展示出来的文本干净清爽
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 切片操作将 assistant 标签之后模型真正生成的台词提取出来
answer = response.split("assistant\n")[-1]

print(f"{answer}\n")
