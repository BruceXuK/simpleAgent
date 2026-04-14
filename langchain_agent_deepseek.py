import time
import os
from dotenv import load_dotenv
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

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
    """计算表达式，支持加减乘除，例如：3*5+2"""
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

# ====================== Prompt + 记忆 ======================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以调用工具并记住上下文，回答简洁准确"),
    MessagesPlaceholder(variable_name="chat_history"),  # 上下文记忆必须加这个
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # 工具调用中间步骤
])

# 记忆存储
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ====================== 构建真正的 Agent ======================
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印详细执行过程
    max_iterations=3  # 防止无限调用
)

# 带记忆的 Agent
agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# ====================== 日志工具 ======================
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

    log("FINAL AI OUTPUT", response["output"])
    return response["output"]


# ====================== 主函数 ======================
if __name__ == '__main__':
    print("🚀 Agent Debug Mode Started (DeepSeek + LangChain)")
    print("输入 exit 退出\n")
    session_id = "user-001"
    while True:
        q = input("你：").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue
        try:
            result = run_agent(session_id, q)
            print("AI：", result, "\n")
        except Exception as e:
            print("❌ 错误：", e)