import time
import os
from dotenv import load_dotenv

# ==========================
# LangChain 核心组件
# ==========================
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI  # OpenAI/DeepSeek 模型封装
from langchain_core.tools import tool     # 用于定义工具函数
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Prompt模板


# ==========================
# 1. 加载 .env 环境变量
# ==========================
# 作用：读取 OPENAI_API_KEY / BASE_URL 等配置
load_dotenv()


# =========================================================
# 2. 定义工具（Tools）
# =========================================================

@tool
def get_current_time() -> str:
    """
    工具1：获取当前时间
    👉 Agent 在需要时间时会自动调用这个函数
    """
    return time.strftime('%Y-%m-%d %H:%M:%S')


@tool
def get_weather(city: str) -> str:
    """
    工具2：天气查询（模拟数据）

    参数:
        city: 城市名称，例如 北京 / 上海 / Singapore

    返回:
        模拟天气结果
    """
    return f"{city} 天气：晴，24℃（模拟数据）"


@tool
def calculator(expr: str) -> str:
    """
    工具3：数学计算器

    参数:
        expr: 数学表达式，例如 3*5+2

    返回:
        计算结果（字符串）
    """
    try:
        return str(eval(expr))  # ⚠️ 生产环境建议替换为安全解析
    except:
        return "计算失败"


# 把所有工具放进列表（供 Agent 使用）
tools = [get_current_time, calculator, get_weather]


# =========================================================
# 3. 初始化大模型（DeepSeek）
# =========================================================
llm = ChatOpenAI(
    model="deepseek-chat",  # 使用 DeepSeek 模型
    base_url=os.getenv("OPENAI_BASE_URL"),  # API 地址（来自 .env）
    temperature=0           # 控制随机性：0=稳定输出
)


# =========================================================
# 4. Prompt 模板（Agent 思维框架）
# =========================================================
prompt = ChatPromptTemplate.from_messages([
    # 系统提示：定义 Agent 角色
    (
        "system",
        "你是一个智能助手，可以调用工具解决问题，并且回答要简洁清晰"
    ),

    # 用户输入占位符
    ("human", "{input}"),

    # Agent 执行过程中用来记录工具调用的“中间变量”
    # 👉 非常重要，否则工具调用会失败
    MessagesPlaceholder("agent_scratchpad"),
])


# =========================================================
# 5. 创建 Agent（核心大脑）
# =========================================================
# 作用：
# 👉 让 LLM 能“理解工具 + 决定是否调用工具”
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# =========================================================
# 6. Memory（记忆系统）
# =========================================================
# 作用：
# 👉 让 Agent 记住用户历史对话
# 👉 实现“多轮对话能力”

# 用 dict 模拟多用户 session 存储
memory_store = {}


def get_memory(session_id: str):
    """
    为每个用户创建独立记忆空间

    参数:
        session_id: 用户唯一标识（比如 user-001）

    返回:
        ConversationBufferMemory 对象
    """
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",  # 历史变量名
            return_messages=True        # 返回 message 对象（不是字符串）
        )
    return memory_store[session_id]


# =========================================================
# 7. Agent 执行器（执行大脑）
# =========================================================
def create_agent_executor(session_id: str):
    """
    创建 Agent 执行器

    👉 每个 session 都有独立 memory
    """
    return AgentExecutor(
        agent=agent,        # 大脑
        tools=tools,        # 工具集
        memory=get_memory(session_id),  # 记忆系统
        verbose=True,       # 打印推理过程（调试非常重要）
        max_iterations=3    # 防止无限循环调用工具
    )


# =========================================================
# 8. 日志系统（调试用）
# =========================================================
def log(title, content):
    """
    打印结构化日志
    """
    print("\n" + "=" * 80)
    print(f"[{title}]")
    print("-" * 80)
    print(content)
    print("=" * 80 + "\n")


# =========================================================
# 9. Agent 执行逻辑
# =========================================================
def run_agent(session_id: str, user_input: str):
    """
    执行一次用户请求

    流程：
    1. 记录输入
    2. 交给 Agent 推理
    3. 自动判断是否调用工具
    4. 返回最终结果
    """

    # 1️⃣ 打印用户输入
    log("USER INPUT", user_input)

    # 2️⃣ 创建 Agent 执行器
    agent_executor = create_agent_executor(session_id)

    # 3️⃣ 调用 Agent
    response = agent_executor.invoke({
        "input": user_input
    })

    # 4️⃣ 打印最终输出
    log("FINAL OUTPUT", response["output"])

    return response["output"]


# =========================================================
# 10. 主程序入口（CLI 对话）
# =========================================================
if __name__ == "__main__":

    print("🚀 LangChain Tool Agent（DeepSeek + Memory + Tools）")
    print("💡 输入 exit 退出系统\n")

    # 模拟一个用户ID（实际可以替换成登录用户ID）
    session_id = "user-001"

    while True:
        # 获取用户输入
        q = input("你：").strip()

        # 退出条件
        if q.lower() == "exit":
            print("👋 已退出系统")
            break

        # 空输入跳过
        if not q:
            continue

        try:
            # 执行 Agent
            result = run_agent(session_id, q)

            # 输出结果
            print("AI：", result, "\n")

        except Exception as e:
            print("❌ 运行错误：", e)