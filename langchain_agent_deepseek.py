import time
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ==========================
# 加载 .env
# ==========================
load_dotenv()


# ==========================
# 工具
# ==========================
@tool
def get_current_time() -> str:
    """获取当前时间"""
    return time.strftime('%Y-%m-%d %H:%M:%S')


@tool
def calculator(expr: str) -> str:
    """计算表达式"""
    try:
        return str(eval(expr))
    except:
        return "计算失败"


@tool
def get_weather(city: str) -> str:
    """天气查询"""
    return f"{city} 天气：晴，24℃（模拟数据）"


tools = [get_current_time, calculator, get_weather]


# ==========================
# LLM
# ==========================
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)


# ==========================
# Prompt
# ==========================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以调用工具并记住上下文"),
    ("human", "{input}")
])

chain = prompt | llm_with_tools


# ==========================
# Memory
# ==========================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


# ==========================
# ⭐ 日志工具（核心新增）
# ==========================
def log(title, content):
    print("\n" + "=" * 80)
    print(f"[{title}]")
    print("-" * 80)
    print(content)
    print("=" * 80 + "\n")


# ==========================
# Agent 执行（带日志）
# ==========================
def run_agent(session_id: str, user_input: str):

    # 1️⃣ 打印用户输入
    log("USER INPUT", user_input)

    # 2️⃣ 构造 Prompt 输入
    payload = {"input": user_input}

    log("SEND TO LLM (INPUT PAYLOAD)", payload)

    # 3️⃣ 调用 LLM
    response = chain_with_memory.invoke(
        payload,
        config={"configurable": {"session_id": session_id}}
    )

    # 4️⃣ 打印 LLM 原始输出
    log("LLM RAW RESPONSE", response)

    # 5️⃣ tool_calls
    if response.tool_calls:
        log("TOOL CALLS", response.tool_calls)

        results = []

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]

            log("EXEC TOOL", f"{tool_name}({args})")

            for t in tools:
                if t.name == tool_name:
                    result = t.invoke(args)

                    log("TOOL RESULT", result)

                    results.append(f"{tool_name}: {result}")

        # 6️⃣ LLM 总结输入
        final_input = f"""
用户问题：
{user_input}

工具返回：
{chr(10).join(results)}

请整理成自然语言回答
"""

        log("SEND TO LLM (FINAL SUMMARY INPUT)", final_input)

        final_response = llm.invoke(final_input)

        log("FINAL LLM OUTPUT", final_response.content)

        return final_response.content

    return response.content


# ==========================
# CLI
# ==========================
if __name__ == "__main__":
    print("🚀 Agent Debug Mode Started (DeepSeek + LangChain)")
    print("输入 exit 退出\n")

    session_id = "user-001"

    while True:
        q = input("你：").strip()

        if q.lower() == "exit":
            break

        try:
            result = run_agent(session_id, q)
            print("AI：", result, "\n")
        except Exception as e:
            print("❌ 错误：", e)