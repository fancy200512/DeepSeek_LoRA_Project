import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B"
INPUT_PATH = "./data/huanhuan.json"  # 建议写相对路径，方便迁移
OUTPUT_PATH = "./data/huanhuan_cot.json"

def main():
    print("加载基座模型与分词器进行数据增强...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.eval()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cot_data = []
    # 可以先截取前 10 条测试，跑通后再全量生成：for item in tqdm(data[:10], desc="生成思维链"):
    for item in tqdm(data, desc="生成思维链"):
    # for item in tqdm(data[:10], desc="生成思维链"):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        original_output = item.get("output", "")
        
        user_content = f"{instruction}\n{input_text}" if input_text else instruction
        think_prompt = (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>"
        )
        
        inputs = tokenizer(think_prompt, return_tensors="pt").to(model.device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150, # 稍微给足一点思考空间
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    eos_token_id=tokenizer.encode("</think>", add_special_tokens=False)[0],
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            
            raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 清理模型输出，确保只提取思考部分
            think_text = raw.replace("</think>", "").strip()
            if not think_text:
                think_text = "此情此景，我需谨慎作答。" # 兜底策略，防止生成空思考
                
            full_output = f"<think>\n{think_text}\n</think>\n<answer>\n{original_output}\n</answer>"
            
            cot_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": full_output
            })
            
        except Exception as e:
            print(f"生成失败跳过当前条目: {e}")
            continue

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cot_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已成功生成 {len(cot_data)} 条带CoT的训练数据 -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()