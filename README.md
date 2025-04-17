
# 推荐系统项目

本项目实现了一个完整的推荐系统，从数据预处理、离线内容分析、协同过滤、排序模型训练，到在线推荐服务（基于 gRPC 接口）均有详细实现。项目架构模块化，便于扩展和维护，支持题库定时更新和实时推荐。

---
## 传统推荐系统与PTFR对比
| **维度**     | **传统推荐系统**          | **PTFR 提升**                                                           |
|--------------|---------------------|------------------------------------------------------------------------|
| **召回**     | 1 ~ 2 个通道（Tag / CF） | **5 通道并行** + 动态开关                           |
| **排序权重** | 固定常量 / 离线调参         | **Redis 热加载**，A/B 或灰度                               |
| **时序热度** | 纯 PV / UV           | 评分 + 浏览 × 指数衰减 + 新题 Boost，兼顾“新”“热”平衡                  |
| **浏览惩罚** | 线性 Views            | `log(Views + 1)` 抑制头部霸榜；                            |
| **多样性**   | Top‑N 截断            | 余弦惩罚 Greedy Re‑rank                       |
| **在线学习** | 每日离线重训              | 增量 CF + Profile 累加                                  |
| **可运维性** | 配置散落                | Data‑Pipeline → KV‑Store → 线上加载           |


## 1. 环境搭建

本项目基于 Python 3.12+ 开发，建议使用虚拟环境管理依赖。

### 1.1 创建虚拟环境

#### 使用 Conda

```bash
# 创建名为 "ReSys" 的新环境，并指定 Python 版本
conda create -n ReSys python=3.12 -y

# 激活环境
conda activate ReSys
```

#### 使用 venv

```bash
# 在项目根目录创建虚拟环境
python -m venv venv

# Windows 下激活虚拟环境
venv\Scripts\activate

# Linux / macOS 下激活虚拟环境
source venv/bin/activate
```

### 1.2 安装依赖

项目所有依赖已整理在 `requirements.txt` 文件中，使用以下命令安装：

```bash
pip install -r requirements.txt
```

#### requirements.txt

```txt
grpcio==1.71.0
grpcio-tools==1.71.0
xgboost==2.1.3
numpy==1.26.4
scikit-learn==1.5.1
transformers==4.45.2
torch==2.5.0
protobuf==5.29.3
pandas==2.2.2
scipy==1.13.1
annoy==1.17.0
```

### 1.3 依赖包说明

- **grpcio, grpcio-tools**  
  用于构建和运行 gRPC 服务，并根据 `.proto` 文件生成 Protobuf 代码。

- **xgboost**  
  用于训练 Learning-to-Rank 排序模型，对候选推荐结果进行精细排序。

- **numpy**  
  提供高效的数值计算和数组操作，是数据处理的基础库。

- **scikit-learn**  
  用于离线内容分析（如 TF-IDF 特征提取、PCA 降维、KMeans 聚类），帮助提取题目文本特征。

- **transformers 和 torch**  
  为 BERT 模型提供支持（可选），用于计算题目的文本嵌入。

- **protobuf**  
  用于数据序列化，与 gRPC 协同工作。

- **pandas**  
  用于数据分析和 CSV 数据预处理。

- **scipy**  
  提供科学计算相关函数，辅助数据处理。

- **annoy**  
  用于构建近似最近邻索引（可选），加速大规模向量检索任务。

*注意：Python 标准库（如 os、sys、csv、json、math、time、datetime、collections、typing、statistics）无需额外安装。*

---

## 2. 模块组成

项目整体结构如下：

```
ReSys/
├── models/
│   ├── data_pipline.py                # 数据预处理：CSV 读取、清洗和转换
│   ├── collaborative_filter.py        # 基于物品协同过滤模型
│   ├── diversity.py                   # 多样性重排算法
│   ├── offline_analysis.py            # 离线内容分析、特征提取、题目聚类和热门度计算
│   ├── ranking_model_training.py      # 排序模型训练（Learning-to-Rank 示例，使用 XGBoost）
│   ├── recommender.py                 # 推荐系统核心逻辑：多通道召回、排序融合与多样性重排
│   ├── simulate_online_recommendation.py  # 模拟在线推荐流程示例
│   ├── user_profile.py                # 用户画像管理与交互数据维护
│   ├── main.py                        # 离线推荐整合测试入口
│   ├── data.csv                       # 数据集
│   ├── question_bank.json             # 通过运行question_bank_updater.py来获得
│   └── question_bank_updater.py       # 定时更新题库：每 5 分钟从 CSV 构建题库并保存为 JSON
│
└── gRPCServer/
    ├── recommendation.proto           # gRPC 接口定义（消息和服务）
    ├── recommendation_pb2.py          # 自动生成的 Protobuf 消息代码
    ├── recommendation_pb2_grpc.py       # 自动生成的 gRPC 接口代码
    ├── recommendation_service.py      # gRPC 服务实现：处理交互上报和推荐请求
    ├── run_server.py                  # 启动 gRPC 服务器
    └── test_client.py                 # 测试客户端：调用交互上报和推荐接口
```

---

## 3. 模块功能介绍

### 3.1 数据预处理 (`models/data_pipline.py`)

- **功能**：  
  读取 CSV 文件，清洗和转换原始数据，将其解析为字典列表，为后续模型训练与推荐提供标准化数据。

### 3.2 协同过滤 (`models/collaborative_filter.py`)

- **功能**：  
  基于用户与题目交互数据构建物品间的相似度矩阵，利用余弦相似度实现 Item-based 协同过滤，用于召回候选题目。

### 3.3 多样性重排 (`models/diversity.py`)

- **功能**：  
  对初步推荐结果进行多样性重排，避免返回的题目过于相似，提升推荐结果的多样性和新鲜度。

### 3.4 离线内容分析 (`models/offline_analysis.py`)

- **功能**：  
  利用 TF-IDF（或 BERT 嵌入）、PCA 降维和 KMeans 聚类，提取题目文本特征，并计算题目热门度（结合评分、浏览量和时间衰减）。

### 3.5 排序模型训练 (`models/ranking_model_training.py`)

- **功能**：  
  使用 XGBoost Ranker 训练 Learning-to-Rank 模型，根据多个特征（如 CF 得分、标签匹配、热门度等）对候选题目进行精细排序。

### 3.6 推荐核心逻辑 (`models/recommender.py`)

- **功能**：  
  结合多通道召回（标签、用户兴趣、协同过滤、热门题和随机探索）和排序融合（含多样性重排），生成最终推荐结果。

### 3.7 用户画像管理 (`models/user_profile.py`)

- **功能**：  
  构建和维护用户画像，记录用户历史交互、兴趣标签和其他特征，为个性化推荐提供数据支持。

### 3.8 题库更新 (`models/question_bank_updater.py`)

- **功能**：  
  每 5 分钟从 CSV 文件中聚合题库数据，经过一定的过滤和聚合后保存为 JSON 文件，在线推荐服务直接从 JSON 文件加载题库数据，解耦数据更新和在线服务。

### 3.9 在线推荐服务（gRPC）

- **gRPC 接口定义**：  
  `gRPCServer/recommendation.proto` 定义了用户交互消息和推荐请求/响应格式，以及 Recommender 服务接口（SendInteraction、Recommend）。

- **服务实现**：  
  `gRPCServer/recommendation_service.py` 实现了上述接口，调用 `models/recommender.py` 中的推荐逻辑，并从 JSON 文件中加载题库数据。

- **服务器启动与测试**：  
  `gRPCServer/run_server.py` 启动服务器，`gRPCServer/test_client.py` 提供测试客户端示例。

---

## 4. 使用方法

### 4.1 数据准备

- 将原始数据 CSV 文件（如 `data.csv`）放在项目根目录（或指定路径）。  
- CSV 文件应包含以下字段（分隔符可为逗号或制表符，但 header 行需正确解析）：  
  - `user_id`
  - `question_id`
  - `timestamp`
  - `views`
  - `rating`
  - `user_interest`
  - `question_keywords`
  - `question_description`

### 4.2 题库更新

在 `models` 目录下运行题库更新程序，程序会每 5 分钟更新一次题库并生成 JSON 文件（`question_bank.json`）：

```bash
cd models
python question_bank_updater.py
```

检查生成的 `question_bank.json` 文件，确认其中包含有效的题库数据。

### 4.3 离线推荐测试

如果需要进行离线测试，可运行 `models/main.py`，该脚本会整合所有模块并输出推荐结果：

```bash
python main.py
```

### 4.4 启动 gRPC 服务

在 `gRPCServer` 目录下启动 gRPC 服务器：

```bash
cd gRPCServer
python run_server.py
```

服务器启动后会自动加载 `models/question_bank.json` 中的题库数据，并监听 50051 端口。

### 4.5 测试客户端

在另一个终端，运行测试客户端，发送用户交互数据并请求推荐结果：

```bash
python test_client.py
```

测试客户端将首先调用 `SendInteraction` 上报交互数据，然后调用 `Recommend` 获取推荐结果，并打印每个推荐题目的详细信息（题目 ID、标签、评分、浏览量、热门度等）。

### 4.6 其他注意事项

- **模块化设计**  
  每个模块均可独立运行和调试，方便逐步完善和扩展推荐算法。

- **运行路径设置**  
  为确保各模块正确导入，请从项目根目录启动各脚本或正确设置 PYTHONPATH。例如，在运行 gRPC 服务时，`run_server.py` 已动态添加项目根目录到 Python 路径中。

- **依赖管理**  
  建议使用虚拟环境隔离依赖，确保不同项目之间不会相互干扰。

---

## 5训练流程
![P‑TFR 架构图](/images/figure.png)
---
## 总结

本项目提供了一整套推荐系统解决方案，涵盖数据预处理、离线分析、协同过滤、排序模型训练、用户画像与题库更新，以及基于 gRPC 的在线推荐服务。模块间解耦、设计清晰，既支持离线批量分析，也支持实时推荐更新。希望本项目能为你构建和扩展个性化推荐系统提供有力支持！

---

