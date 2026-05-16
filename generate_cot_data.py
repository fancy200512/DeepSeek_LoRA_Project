import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B"  # 本地基座模型的路径
INPUT_PATH = "./data/huanhuan.json"          # 原始对话数据路径
OUTPUT_PATH = "./data/huanhuan_cot.json"     # 生成后的带思维链的数据集保存路径

def main():
    print("加载基座模型与分词器进行数据增强...")
    
    # 加载分词器 (Tokenizer)
    # trust_remote_code=True: 允许执行模型仓库中自定义的Python代码
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # 加载LLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        # 指定模型加载的精度。bfloat16 相比float32能节省一半显存，且在Ampere架构显卡上训练更稳定，不易发生梯度溢出
        torch_dtype=torch.bfloat16, 
        #自动将模型的不同层分配到当前可用的GPU显存或CPU内存上
        device_map="auto", 
        trust_remote_code=True
    )
    # 将模型设置为评估模式 ，关闭Dropout等训练专属机制，保证推理结果的稳定性
    model.eval()

    # 读取原始数据集
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cot_data = []
    
    # tqdm 用于在终端显示进度条，方便监控几万条数据的处理进度
    for item in tqdm(data, desc="生成思维链"):
        
        # 提取原始数据中的三个核心字段：指令、用户输入、原本的输出台词
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        original_output = item.get("output", "")
        
        # 拼接用户提问内容
        user_content = f"{instruction}\n{input_text}" if input_text else instruction
        
        # 构建推理引导 Prompt
        # 这里使用了ChatML格式，也是很多开源模型的默认对话模板）。
        # 在 prompt 末尾强行加上了 <think> 标签
        think_prompt = (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>"
        )
        
        # 将文本 Prompt 转换为张量并发送到模型所在的 GPU 设备上
        inputs = tokenizer(think_prompt, return_tensors="pt").to(model.device)
        
        try:
            # torch.no_grad(): 告诉 PyTorch 当前处于推理阶段，不需要计算和存储梯度，可以大幅节约显存加速推理
            with torch.no_grad():
                # 调用模型生成核心参数
                outputs = model.generate(
                    #限制模型最多生成多少个新的 token（字/词）
                    max_new_tokens=150, 
                    # 采样温度（0.0 到 1.0+）。越低模型越死板，越高模型越具备创造力。0.7 是生成类任务的常用均衡值
                    temperature=0.7,
                    # 开启概率采样。如果不开启，模型永远只会输出概率最高的那句话（贪心搜索），容易死循环或复读机
                    do_sample=True,
                    # 核采样: 配合 temperature 使用，0.9 代表只在累计概率达到 90% 的头部候选词中进行随机采样，避免生成完全无关的乱码
                    top_p=0.9,
                    # 结束符，这里设定当模型吐出 "</think>" 的第一个 token 时，就立刻停止生成，防止它瞎编后面的回答
                    eos_token_id=tokenizer.encode("</think>", add_special_tokens=False)[0],
                    # 填充符，如果批量推理时长度不一用来补齐空缺，为了兼容性通常将其设为 eos_token_id
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            
            # 输出解码与后处理
            # inputs.input_ids.shape[1]: 切片操作是为了把我们原本喂给模型的 prompt 截断掉，只保留模型全新生成的思考内容。
            # skip_special_tokens=True: 解码时自动过滤掉类似 <|im_end|> 这种特殊控制符，让文本更干净。
            raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 文本清洗：防止模型把提前停止的触发符 </think> 也输出了，手动抹掉并去掉首尾空格
            think_text = raw.replace("</think>", "").strip()
            
            # 异常处理：有时候模型抽风生成了空白，如果不兜底，训练时就会报格式错误
            if not think_text:
                think_text = "此情此景，我需谨慎作答。" # 兜底策略，给一个通用的万金油心理活动
                
            # 拼装双段式语料
            # 将模型生成的心理活动，与原数据中真实的甄嬛台词拼接起来，形成双段式数据结构
            full_output = f"<think>\n{think_text}\n</think>\n<answer>\n{original_output}\n</answer>"
            
            # 将新构造的数据存入列表
            cot_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": full_output
            })
            
        except Exception as e:
            # 捕获因显存不足或张量维度错误导致的崩溃，跳过坏数据，保障整个跑库流程不中断
            print(f"生成失败跳过当前条目: {e}")
            continue

    # 保存最终结果
    # ensure_ascii=False: 处理中文JSON，如果不加，所有的中文字符会被存为\uXXXX 的形式，变成乱码。
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cot_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 已成功生成 {len(cot_data)} 条带 CoT 的训练数据 -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
