from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import re

app = FastAPI(title="本地函数调用模型服务")

# 4bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_path = "./Qwen2.5-3B-Instruct"

print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print("✅ 模型加载完成！")

# 请求结构体
class ChatRequest(BaseModel):
    prompt: str
    functions: list

# 解析工具调用格式
def parse_function_call(text):
    try:
        text = text.strip()
        func_match = re.search(r"FUNCTION:\s*(\w+)", text)
        params_match = re.search(r"PARAMS:\s*(\{.*\})", text)

        if not func_match or not params_match:
            return None

        func_name = func_match.group(1).strip()
        params = json.loads(params_match.group(1).strip())
        return {"name": func_name, "parameters": params}
    except:
        return None

# 对话接口
@app.post("/agent/chat")
def agent_chat(request: ChatRequest):
    user_q = request.prompt
    functions = request.functions

    # ======================
    # 【强约束提示词】
    # 严格禁止模型自己回答！必须输出工具调用！
    # ======================
    system_prompt = f"""
你只能做一件事：选择工具并按格式输出。
你**绝对不能自己回答问题**，不能编造结果！

可用工具：
{json.dumps(functions, ensure_ascii=False, indent=2)}

必须严格只输出以下格式：
FUNCTION: 函数名
PARAMS: {{"key": "value"}}

不许回答问题！不许解释！只许输出调用格式！
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_q}
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.01
    )

    res = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    func = parse_function_call(res)

    if func:
        return {
            "type": "function_call",
            "name": func["name"],
            "parameters": func["parameters"]
        }
    else:
        # 模型不听话自己编了，强制返回空文本
        return {"type": "text", "content": ""}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)