import time
import os
from dotenv import load_dotenv
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# 加载 .env
load_dotenv()

# ====================== 工具定义 ======================
@tool
def get_current_time() -> str:
    """获取当前时间"""
    return time.strftime('%Y-%m-%d %H:%M:%S')

@tool
def get_weather(city: str) -> str:
    """天气查询"""
    return f"{city} 天气：晴，24℃（模拟数据）"

@tool
def calculator(expr: str) -> str:
    """计算表达式"""
    try:
        return str(eval(expr))
    except:
        return "计算失败"

tools = [get_current_time, calculator, get_weather]

# ====================== LLM 模型 ======================
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)

# ====================== 历史压缩核心函数 ======================
def compress_chat_history(messages: list, keep_recent=4, max_summary_length=800):
    """
    压缩聊天历史：保留最近 keep_recent 条，更早的压缩成摘要
    """
    if len(messages) <= keep_recent:
        return messages  # 不够长，不压缩

    # 拆分：旧历史 + 最近历史
    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    # 把旧消息转成文本
    text_history = "\n".join([
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in old_messages
    ])

    # 让 LLM 压缩
    compress_prompt = f"""
请把以下对话历史压缩成一段简洁摘要，保留关键信息（用户问题、结论、工具结果），不要冗余：
{text_history}
摘要（不超过{max_summary_length}字）：
"""
    summary = llm.invoke(compress_prompt).content.strip()

    # 构造新历史：1条摘要消息 + 最近完整消息
    compressed_messages = [
        AIMessage(content=f"【历史摘要】\n{summary}")
    ] + recent_messages

    return compressed_messages

# ====================== 带压缩的记忆存储 ======================
store = {}

def get_session_history_with_compress(session_id: str, keep_recent=4):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    history = store[session_id]
    # 压缩
    history.messages = compress_chat_history(history.messages, keep_recent=keep_recent)
    return history

# ====================== Prompt ======================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以调用工具并记住上下文"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ====================== Agent ======================
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    max_iterations=3
)

# 绑定带压缩的记忆
agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    lambda sid: get_session_history_with_compress(sid, keep_recent=4),  # 压缩入口
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ====================== 日志 ======================
def log(title, content):
    print("\n" + "=" * 80)
    print(f"[{title}]")
    print("-" * 80)
    print(content)
    print("=" * 80 + "\n")

# ====================== Agent 执行 ======================
def run_agent(session_id: str, user_input: str):
    log("USER INPUT", user_input)
    response = agent_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    log("FINAL OUTPUT", response["output"])
    return response["output"]

# ====================== 主函数 ======================
if __name__ == '__main__':
    print("🚀 Agent with Chat History Compression (DeepSeek + LangChain)")
    print("输入 exit 退出\n")
    session_id = "user-001"
    while True:
        q = input("你：").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue
        try:
            res = run_agent(session_id, q)
            print("AI：", res, "\n")
        except Exception as e:
            print("❌ 错误：", e)