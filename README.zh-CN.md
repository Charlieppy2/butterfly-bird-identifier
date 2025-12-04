# HK Urban Ecological Identification System

香港城市生态识别系统 - 一个基于深度学习的蝴蝶与鸟类识别Web应用

## 项目简介

这是一个使用深度学习技术开发的Web应用系统，用于识别香港城市中的蝴蝶与鸟类。系统采用迁移学习技术，基于MobileNetV2构建分类模型，能够识别300+种物种（200种鸟类 + 100+种蝴蝶/蛾类），并提供友好的Web界面供用户上传图片进行识别。

## ✨ 主要功能

### 🔍 核心识别功能
- **图片上传识别**: 支持拖拽上传或选择文件（PNG, JPG, JPEG, GIF, WEBP）
- **实时拍摄**: 使用设备摄像头拍摄图片进行识别
- **批量识别**: 一次上传多张图片进行批量识别
- **智能识别**: 基于深度学习的图像分类，提供Top-3预测结果和置信度

### 📊 图片质量分析
- **多维度分析**: 亮度、对比度、清晰度、饱和度、分辨率
- **质量评分**: 总体质量分数（0-100）
- **智能建议**: 根据图片质量问题提供改进建议

### 💬 AI聊天助手
- **智能问答**: 回答关于物种识别、观察技巧等问题
- **知识库**: 包含栖息地、观察时间、拍照技巧等信息
- **可训练**: 支持扩展和训练AI助手的知识库

### 📈 统计分析
- **识别历史统计**: 总识别次数、独特物种数、平均置信度
- **类别分布**: 鸟类、蝴蝶/蛾类的分布统计
- **置信度分布**: 高/中/低置信度的分布图表
- **Top物种**: 最常识别的物种排行榜
- **时间分布**: 识别活动的时间趋势

### ❤️ 收藏功能
- **收藏物种**: 一键收藏感兴趣的识别结果
- **收藏管理**: 查看、管理所有收藏的物种
- **数据持久化**: 使用localStorage保存收藏数据

### 📜 历史记录
- **识别历史**: 自动保存最近的识别记录
- **快速查看**: 快速浏览历史识别结果
- **标签切换**: 在历史记录和收藏之间轻松切换

## 技术栈

### 模型训练
- **TensorFlow/Keras**: 深度学习框架
- **MobileNetV2**: 预训练模型（迁移学习）
- **Python 3.8+**: 编程语言
- **OpenCV**: 图像处理和质量分析

### Web应用
- **前端**: React 18.2.0
  - Axios: HTTP客户端
  - 响应式设计，支持移动端
- **后端**: Flask 3.0.0
  - Flask-CORS: 跨域支持
  - TensorFlow: 模型推理
  - PIL/OpenCV: 图像处理

## 项目结构

```
butterfly-bird-identifier/
├── data/
│   ├── raw/              # 原始数据集
│   ├── processed/        # 处理后的数据（train/val/test）
│   └── dataset_info.txt  # 数据集信息
├── models/
│   ├── training/         # 训练脚本
│   │   ├── train_model.py      # 模型训练
│   │   ├── prepare_data.py     # 数据准备
│   │   ├── test_model.py       # 模型测试
│   │   └── check_training.py   # 训练进度检查
│   └── trained/          # 训练好的模型
│       ├── model.h5           # 训练好的模型（使用Git LFS）
│       └── class_names.json   # 类别名称列表
├── web_app/
│   ├── frontend/         # React前端应用
│   │   ├── src/
│   │   │   ├── App.js         # 主应用组件
│   │   │   ├── App.css        # 样式文件
│   │   │   └── index.js       # 入口文件
│   │   ├── public/
│   │   │   └── index.html     # HTML模板
│   │   └── package.json       # 前端依赖
│   ├── backend/          # Flask后端API
│   │   ├── app.py             # Flask应用主文件
│   │   ├── requirements.txt   # Python依赖
│   │   ├── knowledge_base.json # AI助手知识库
│   │   └── train_assistant.py  # AI助手训练脚本
│   └── preview.html      # 预览页面
├── notebooks/            # Jupyter notebooks（数据探索）
├── report/              # 项目报告
├── .gitattributes        # Git LFS配置
├── .gitignore           # Git忽略文件
└── README.md            # 本文件
```

## 🚀 快速开始

### 前置要求

1. **Python 3.8+**
   - 下载地址：https://www.python.org/downloads/
   - 安装时请勾选 "Add Python to PATH"

2. **Node.js 16+**
   - 下载地址：https://nodejs.org/
   - 建议安装 LTS 版本

3. **Git LFS** (用于下载大文件)
   ```bash
   git lfs install
   ```

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/Charlieppy2/butterfly-bird-identifier.git
cd butterfly-bird-identifier
```

#### 2. 安装后端依赖

```bash
cd web_app/backend
pip install -r requirements.txt
```

#### 3. 安装前端依赖

```bash
cd ../frontend
npm install
```

### 启动应用

#### 方法一：手动启动（推荐）

**启动后端服务：**

```bash
cd web_app/backend
python app.py
```

后端服务将在 `http://localhost:5000` 启动

**启动前端应用：**

打开新的终端窗口：

```bash
cd web_app/frontend
npm start
```

前端应用将在 `http://localhost:3000` 启动，浏览器会自动打开。

#### 方法二：使用批处理文件（Windows）

**后端：**
```bash
cd web_app/backend
start_backend.bat
```

**前端：**
```bash
cd web_app/frontend
start_frontend.bat
```

## 📖 使用指南

### 识别物种

1. **上传图片**：
   - 点击 "Choose File" 按钮选择图片
   - 或直接拖拽图片到上传区域

2. **拍摄图片**：
   - 点击 "📷 Use Camera" 按钮
   - 允许浏览器访问摄像头
   - 点击 "📸 Capture" 拍摄

3. **查看结果**：
   - 系统会显示识别结果和置信度
   - 显示Top-3预测结果
   - 自动进行图片质量分析

### 使用AI助手

1. 点击右下角的聊天图标打开AI助手
2. 可以询问：
   - 识别技巧
   - 最佳观察时间
   - 拍照建议
   - 物种信息

### 查看统计

1. 在历史记录区域点击 "📊 View Statistics"
2. 查看：
   - 总识别次数
   - 类别分布（鸟类/蝴蝶）
   - 置信度分布
   - Top识别物种

### 收藏功能

1. **收藏物种**：
   - 识别完成后，点击结果标题旁的❤️按钮

2. **查看收藏**：
   - 点击 "❤️ Favorites" 标签
   - 查看所有收藏的物种

3. **移除收藏**：
   - 在收藏列表中点击 "❌ Remove" 按钮
   - 或再次点击❤️按钮取消收藏

## 🎓 模型训练

### 数据准备

将原始图片按类别组织到 `data/raw/` 目录下：

```
data/raw/
├── 001.Black_footed_Albatross/
│   ├── image1.jpg
│   └── ...
├── 002.Laysan_Albatross/
│   └── ...
└── ...
```

运行数据准备脚本：

```bash
cd models/training
python prepare_data.py
```

### 训练模型

```bash
cd models/training
python train_model.py
```

训练参数可在 `train_model.py` 中调整：
- `IMAGE_SIZE`: 图片尺寸 (224, 224)
- `BATCH_SIZE`: 批次大小 (32)
- `EPOCHS`: 训练轮数 (100)
- `LEARNING_RATE`: 学习率 (0.0001)

训练完成后，模型将保存在 `models/trained/model.h5`

### 检查训练进度

```bash
cd models/training
python check_training.py
```

### 测试模型

```bash
cd models/training
python test_model.py
```

## 🤖 训练AI助手

详细指南请参考：[如何訓練AI助手.md](如何訓練AI助手.md)

快速开始：

```bash
cd web_app/backend
python train_assistant.py
```

## 📊 数据集信息

- **总类别数**: 301种（200种鸟类 + 101种蝴蝶/蛾类）
- **数据增强**: 旋转、翻转、缩放、亮度调整
- **图片尺寸**: 224x224
- **训练/验证/测试**: 自动划分

## 🔧 API端点

### 后端API

- `GET /` - 健康检查
- `GET /api/health` - 模型状态
- `GET /api/classes` - 获取所有类别名称
- `POST /api/predict` - 图片识别
- `POST /api/analyze-quality` - 图片质量分析
- `POST /api/statistics` - 获取统计数据
- `POST /api/chat` - AI聊天助手

## 🛠️ 开发环境

- Python 3.8+
- Node.js 16+
- TensorFlow 2.15.0+
- React 18.2.0
- Flask 3.0.0
- OpenCV 4.8.0+

## ⚠️ 注意事项

1. **Git LFS**: 模型文件使用Git LFS存储，克隆后需要运行 `git lfs install`
2. **首次运行**: 首次运行需要加载模型，可能需要一些时间
3. **GPU加速**: 训练模型建议使用GPU加速（Google Colab推荐）
4. **磁盘空间**: 确保有足够的磁盘空间存储数据集和模型（模型约19MB）
5. **浏览器兼容性**: 建议使用Chrome、Firefox或Edge最新版本

## 📝 更新日志

### v1.0.0 (最新)
- ✨ 新增收藏功能
- ✨ 新增图片质量分析
- ✨ 新增AI聊天助手
- ✨ 新增识别历史统计和分析
- ✨ 新增批量识别模式
- 🐛 修复分类分布问题（蝴蝶正确分类）
- 📦 使用Git LFS管理大文件

## 📚 参考资料

- [TensorFlow官方文档](https://www.tensorflow.org/)
- [Keras迁移学习指南](https://keras.io/guides/transfer_learning/)
- [React官方文档](https://react.dev/)
- [Flask官方文档](https://flask.palletsprojects.com/)
- [Git LFS文档](https://git-lfs.github.com/)

## 📄 授权

本项目仅用于学术和教育目的。

## 👥 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

**注意**: 请确保在提交前完成所有必要的配置和测试。

## 📄 其他语言版本

- [English Version](README.md)
- [繁體中文版 (Traditional Chinese)](README.zh-TW.md)

