# 本地智能体 Agent（精简版）
## 项目简介
基于 Qwen2.5-3B-Instruct 本地大模型，实现多工具混合调用的轻量级智能体，纯本地部署，低显存可运行。

## 项目能做什么
- 查询城市天气
- 获取当前时间
- 进行基础数学计算（+、-、*、/）
- 支持多需求混合查询（如“天气+时间+计算”）

## 核心功能
- 多工具混合调用，自动调度、汇总结果
- 自动去重，避免重复调用，不无限循环
- 4bit 量化，4GB 显存即可稳定运行
- 工具可快速扩展，无需修改核心逻辑
- 纯本地运行，不依赖外部 API

## 项目结构
```
simlpeAgent/
├── Qwen2.5-3B-Instruct  # 模型文件
├── api_server.py        # 模型服务端，提供工具调用接口
├── agent_demo.py        # 客户端，负责交互、工具执行与调度
├── requirements.txt     # 项目依赖清单
└── README.md            # 项目说明文档
```

## 启动步骤（简洁版）
1.  先安装依赖（项目根目录执行）：
    ```bash
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    ```
2.  启动模型服务端：
    ```bash
    python api_server.py
    ```
3.  新打开控制台，启动客户端：
    ```bash
    python agent_demo.py
    ```
4.  输入问题即可使用，输入`exit`退出。