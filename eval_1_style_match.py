import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ================= 参数配置 =================
# 本地预训练模型路径
MODEL_ID = "/root/autodl-tmp/DeepSeek-1.5B"
BASELINE_LORA_PATH = "./output/deepseek-lora-train-out/final" 
OURS_LORA_PATH = "./output/het_rank_focal_out/final"
# 结果输出
OUTPUT_CSV = "eval_results.csv"

# 甄嬛专属风格词典
STYLE_WORDS = ["本宫", "臣妾", "皇上", "真真", "左右不过", "福气", "莞尔", "惠贵人"]

# 带有“真实台词 (Ground Truth)”的基础测试集（用于快速验证角色身份与基本常识）
TEST_DATA = [
    {
        "question": "你是谁？",
        "truth": "臣妾甄嬛，家父是大理寺少卿甄远道。"
    },
    {
        "question": "你现在住在哪个宫里？",
        "truth": "回您的话，臣妾现下住在碎玉轩。那里虽然偏远了些，倒也算清静。"
    },
    {
        "question": "你平常喜欢做些什么打发时间？",
        "truth": "臣妾闲暇时也就是在宫里看看书、品品茶，或者和槿汐一起做些女红罢了。"
    },
    {
        "question": "今儿个天气真好。",
        "truth": "是啊，外头风和日丽的，真真是个好天气。若是能去御花园走走，赏赏花，自然是极好的。"
    },
    {
        "question": "皇上驾到！",
        "truth": "臣妾给皇上请安，愿皇上万福金安。"
    }
]

print("正在加载基座模型 (使用 bfloat16)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

def generate_answer(model, question, use_cot=False):
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{'<think>' if use_cot else ''}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=200, temperature=0.7, repetition_penalty=1.1,
            eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    if use_cot:
        if "<answer>" in full_text and "</answer>" in full_text:
            return full_text.split("<answer>")[-1].split("</answer>")[0].strip()
        elif "</think>" in full_text:
            return full_text.split("</think>")[-1].replace("<answer>", "").strip()
    return full_text.strip()

def calculate_style_score(text):
    return sum(1 for word in STYLE_WORDS if word in text)

results = []

# --- 运行 Baseline ---
print("\n--- 运行 Baseline 模型 ---")
model = PeftModel.from_pretrained(base_model, BASELINE_LORA_PATH)
model.eval()
for item in tqdm(TEST_DATA, desc="Baseline 推理"):
    item["baseline_ans"] = generate_answer(model, item["question"], use_cot=False)
model.unload()
del model
torch.cuda.empty_cache()

# --- 运行 Ours ---
print("\n--- 运行 Ours 模型 ---")
model = PeftModel.from_pretrained(base_model, OURS_LORA_PATH)
model.eval()
for item in tqdm(TEST_DATA, desc="Ours 推理"):
    item["ours_ans"] = generate_answer(model, item["question"], use_cot=True)

# --- 统计与保存 ---
b_total, o_total = 0, 0
for item in TEST_DATA:
    b_score = calculate_style_score(item["baseline_ans"])
    o_score = calculate_style_score(item["ours_ans"])
    b_total += b_score
    o_total += o_score
    results.append({
        "Question": item["question"],
        "Ground Truth": item["truth"],
        "Baseline Answer": item["baseline_ans"],
        "Baseline Style": b_score,
        "Ours Answer": item["ours_ans"],
        "Ours Style": o_score
    })

print(f"\n📊 风格词总命中数 -> Baseline: {b_total} | Ours: {o_total}")
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"📁 结果已保存至 {OUTPUT_CSV}")
