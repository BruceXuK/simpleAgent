# 导入需要的库
import requests
import time
import json

# 本地模型接口地址
API_URL = "http://127.0.0.1:8000/agent/chat"

# ======================
# 工具函数
# ======================
def get_current_time():
    """获取当前时间"""
    return f"当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"

def calculator(a: float, b: float, op: str):
    """计算器"""
    if op == "+":
        res = a + b
    elif op == "-":
        res = a - b
    elif op == "*":
        res = a * b
    elif op == "/":
        res = a / b if b != 0 else "不能除以零"
    else:
        return "不支持的运算"
    return f"计算结果：{a}{op}{b}={res}"

def get_weather(city: str):
    """查询天气"""
    return f"{city}天气：晴天 24℃"

# 工具映射
TOOL_MAP = {
    "get_current_time": get_current_time,
    "calculator": calculator,
    "get_weather": get_weather
}

# 工具描述
FUNCTIONS = [
    {
        "name": "get_current_time",
        "description": "获取当前时间",
        "parameters": {}
    },
    {
        "name": "calculator",
        "description": "数学计算",
        "parameters": {
            "a": "number",
            "b": "number",
            "op": "string"
        }
    },
    {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {"city": "string"}
    }
]

# ======================
# 单次工具调用（内部用）
# ======================
def call_once(prompt):
    try:
        resp = requests.post(API_URL, json={
            "prompt": prompt,
            "functions": FUNCTIONS
        }, timeout=10).json()
        return resp
    except:
        return None

# ======================
# 智能体：强制多工具执行，不会漏
# ======================
# def run_agent(user_query):
#     print(f"\n🧑 用户：{user_query}")
#     results = []
#     called = set()
#
#     # 最多执行 6 轮，足够覆盖 天气+时间+计算
#     for i in range(6):
#         print(f"\n===== 第 {i+1} 轮判断 =====")
#
#         # 强制提示模型：还有任务没做完
#         prompt = f"""
# 用户问题：{user_query}
# 已经得到的信息：{results}
# 请继续完成剩下的任务，严格按格式调用工具。
# """
#
#         resp = call_once(prompt)
#         if not resp:
#             print("❌ 请求失败")
#             break
#
#         print("📥 模型返回：", json.dumps(resp, ensure_ascii=False, indent=2))
#
#         # 模型输出文本，直接结束
#         if resp.get("type") == "text":
#             print("\n🎉 回答：", resp["content"])
#             return
#
#         func_name = resp.get("name")
#         params = resp.get("parameters", {})
#
#         # 去重
#         key = func_name + json.dumps(params, sort_keys=True)
#         if key in called:
#             print("🔁 重复调用，跳过")
#             # 多等一轮，避免小模型卡住
#             if i > 3:
#                 break
#             continue
#
#         # 执行工具
#         try:
#             res = TOOL_MAP[func_name](**params)
#             print(f"✅ 结果：{res}")
#             results.append(res)
#             called.add(key)
#         except Exception as e:
#             print(f"❌ 执行失败：{e}")
#             break
#
#     # 最终汇总
#     if results:
#         print("\n🎉 最终回答：" + "，".join(results))
#     else:
#         print("\n🎉 无结果")
def run_agent(user_query):
    print(f"\n🧑 用户：{user_query}")
    results = []

    # --------------
    # 1. 判断是否需要查天气
    # --------------
    if any(key in user_query for key in ["天气", "气温", "温度"]):
        print("\n===== 调用天气工具 =====")
        resp = call_once("提取问题中的城市名")
        if resp and resp["name"] == "get_weather":
            try:
                res = TOOL_MAP[resp["name"]](**resp["parameters"])
                results.append(res)
                print(f"✅ {res}")
            except:
                pass

    # --------------
    # 2. 判断是否需要查时间
    # --------------
    if any(key in user_query for key in ["时间", "几点", "日期"]):
        print("\n===== 调用时间工具 =====")
        res = get_current_time()
        results.append(res)
        print(f"✅ {res}")

    # --------------
    # 3. 判断是否需要计算
    # --------------
    if any(key in user_query for key in ["+", "-", "*", "/", "计算", "多少"]):
        print("\n===== 调用计算工具 =====")
        resp = call_once(user_query)
        if resp and resp["name"] == "calculator":
            try:
                res = TOOL_MAP[resp["name"]](**resp["parameters"])
                results.append(res)
                print(f"✅ {res}")
            except:
                pass

    # 最终输出
    if results:
        print("\n🎉 最终回答：" + "，".join(results))
    else:
        print("\n🎉 最终回答：无法识别需求")

# ======================
# 入口
# ======================
if __name__ == "__main__":
    print("🚀 多工具智能体已启动（天气+时间+计算）")
    while True:
        user = input("\n请输入问题：").strip()
        if user.lower() == "exit":
            break
        run_agent(user)