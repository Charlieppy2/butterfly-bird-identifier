# HK Urban Ecological Identification System

香港城市生態識別系統 - 一個基於深度學習的蝴蝶與鳥類識別Web應用

## 項目簡介

這是一個使用深度學習技術開發的Web應用系統，用於識別香港城市中的蝴蝶與鳥類。系統採用遷移學習技術，基於MobileNetV2構建分類模型，能夠識別300+種物種（200種鳥類 + 100+種蝴蝶/蛾類），並提供友好的Web界面供用戶上傳圖片進行識別。

## ✨ 主要功能

### 🔍 核心識別功能
- **圖片上傳識別**: 支持拖拽上傳或選擇文件（PNG, JPG, JPEG, GIF, WEBP）
- **實時拍攝**: 使用設備攝像頭拍攝圖片進行識別
- **批量識別**: 一次上傳多張圖片進行批量識別
- **智能識別**: 基於深度學習的圖像分類，提供Top-3預測結果和置信度

### 📊 圖片質量分析
- **多維度分析**: 亮度、對比度、清晰度、飽和度、分辨率
- **質量評分**: 總體質量分數（0-100）
- **智能建議**: 根據圖片質量問題提供改進建議

### 💬 AI聊天助手
- **智能問答**: 回答關於物種識別、觀察技巧等問題
- **知識庫**: 包含棲息地、觀察時間、拍照技巧等信息
- **可訓練**: 支持擴展和訓練AI助手的知識庫

### 📈 統計分析
- **識別歷史統計**: 總識別次數、獨特物種數、平均置信度
- **類別分布**: 鳥類、蝴蝶/蛾類的分布統計
- **置信度分布**: 高/中/低置信度的分布圖表
- **Top物種**: 最常識別的物種排行榜
- **時間分布**: 識別活動的時間趨勢

### ❤️ 收藏功能
- **收藏物種**: 一鍵收藏感興趣的識別結果
- **收藏管理**: 查看、管理所有收藏的物種
- **數據持久化**: 使用localStorage保存收藏數據

### 📜 歷史記錄
- **識別歷史**: 自動保存最近的識別記錄
- **快速查看**: 快速瀏覽歷史識別結果
- **標籤切換**: 在歷史記錄和收藏之間輕鬆切換

## 技術棧

### 模型訓練
- **TensorFlow/Keras**: 深度學習框架
- **MobileNetV2**: 預訓練模型（遷移學習）
- **Python 3.8+**: 編程語言
- **OpenCV**: 圖像處理和質量分析

### Web應用
- **前端**: React 18.2.0
  - Axios: HTTP客戶端
  - 響應式設計，支持移動端
- **後端**: Flask 3.0.0
  - Flask-CORS: 跨域支持
  - TensorFlow: 模型推理
  - PIL/OpenCV: 圖像處理

## 項目結構

```
butterfly-bird-identifier/
├── data/
│   ├── raw/              # 原始數據集
│   ├── processed/        # 處理後的數據（train/val/test）
│   └── dataset_info.txt  # 數據集信息
├── models/
│   ├── training/         # 訓練腳本
│   │   ├── train_model.py      # 模型訓練
│   │   ├── prepare_data.py     # 數據準備
│   │   ├── test_model.py       # 模型測試
│   │   └── check_training.py   # 訓練進度檢查
│   └── trained/          # 訓練好的模型
│       ├── model.h5           # 訓練好的模型（使用Git LFS）
│       └── class_names.json   # 類別名稱列表
├── web_app/
│   ├── frontend/         # React前端應用
│   │   ├── src/
│   │   │   ├── App.js         # 主應用組件
│   │   │   ├── App.css        # 樣式文件
│   │   │   └── index.js       # 入口文件
│   │   ├── public/
│   │   │   └── index.html     # HTML模板
│   │   └── package.json       # 前端依賴
│   ├── backend/          # Flask後端API
│   │   ├── app.py             # Flask應用主文件
│   │   ├── requirements.txt   # Python依賴
│   │   ├── knowledge_base.json # AI助手知識庫
│   │   └── train_assistant.py  # AI助手訓練腳本
│   └── preview.html      # 預覽頁面
├── notebooks/            # Jupyter notebooks（數據探索）
├── report/              # 項目報告
├── .gitattributes        # Git LFS配置
├── .gitignore           # Git忽略文件
└── README.md            # 本文件
```

## 🚀 快速開始

### 前置要求

1. **Python 3.8+**
   - 下載地址：https://www.python.org/downloads/
   - 安裝時請勾選 "Add Python to PATH"

2. **Node.js 16+**
   - 下載地址：https://nodejs.org/
   - 建議安裝 LTS 版本

3. **Git LFS** (用於下載大文件)
   ```bash
   git lfs install
   ```

### 安裝步驟

#### 1. 克隆倉庫

```bash
git clone https://github.com/Charlieppy2/butterfly-bird-identifier.git
cd butterfly-bird-identifier
```

#### 2. 安裝後端依賴

```bash
cd web_app/backend
pip install -r requirements.txt
```

#### 3. 安裝前端依賴

```bash
cd ../frontend
npm install
```

### 啟動應用

#### 方法一：手動啟動（推薦）

**啟動後端服務：**

```bash
cd web_app/backend
python app.py
```

後端服務將在 `http://localhost:5000` 啟動

**啟動前端應用：**

打開新的終端窗口：

```bash
cd web_app/frontend
npm start
```

前端應用將在 `http://localhost:3000` 啟動，瀏覽器會自動打開。

#### 方法二：使用批處理文件（Windows）

**後端：**
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

### 識別物種

1. **上傳圖片**：
   - 點擊 "Choose File" 按鈕選擇圖片
   - 或直接拖拽圖片到上傳區域

2. **拍攝圖片**：
   - 點擊 "📷 Use Camera" 按鈕
   - 允許瀏覽器訪問攝像頭
   - 點擊 "📸 Capture" 拍攝

3. **查看結果**：
   - 系統會顯示識別結果和置信度
   - 顯示Top-3預測結果
   - 自動進行圖片質量分析

### 使用AI助手

1. 點擊右下角的聊天圖標打開AI助手
2. 可以詢問：
   - 識別技巧
   - 最佳觀察時間
   - 拍照建議
   - 物種信息

### 查看統計

1. 在歷史記錄區域點擊 "📊 View Statistics"
2. 查看：
   - 總識別次數
   - 類別分布（鳥類/蝴蝶）
   - 置信度分布
   - Top識別物種

### 收藏功能

1. **收藏物種**：
   - 識別完成後，點擊結果標題旁的❤️按鈕

2. **查看收藏**：
   - 點擊 "❤️ Favorites" 標籤
   - 查看所有收藏的物種

3. **移除收藏**：
   - 在收藏列表中點擊 "❌ Remove" 按鈕
   - 或再次點擊❤️按鈕取消收藏

## 🎓 模型訓練

### 數據準備

將原始圖片按類別組織到 `data/raw/` 目錄下：

```
data/raw/
├── 001.Black_footed_Albatross/
│   ├── image1.jpg
│   └── ...
├── 002.Laysan_Albatross/
│   └── ...
└── ...
```

運行數據準備腳本：

```bash
cd models/training
python prepare_data.py
```

### 訓練模型

```bash
cd models/training
python train_model.py
```

訓練參數可在 `train_model.py` 中調整：
- `IMAGE_SIZE`: 圖片尺寸 (224, 224)
- `BATCH_SIZE`: 批次大小 (32)
- `EPOCHS`: 訓練輪數 (100)
- `LEARNING_RATE`: 學習率 (0.0001)

訓練完成後，模型將保存在 `models/trained/model.h5`

### 檢查訓練進度

```bash
cd models/training
python check_training.py
```

### 測試模型

```bash
cd models/training
python test_model.py
```

## 🤖 訓練AI助手

詳細指南請參考：[如何訓練AI助手.md](如何訓練AI助手.md)

快速開始：

```bash
cd web_app/backend
python train_assistant.py
```

## 📊 數據集信息

- **總類別數**: 301種（200種鳥類 + 101種蝴蝶/蛾類）
- **數據增強**: 旋轉、翻轉、縮放、亮度調整
- **圖片尺寸**: 224x224
- **訓練/驗證/測試**: 自動劃分

## 🔧 API端點

### 後端API

- `GET /` - 健康檢查
- `GET /api/health` - 模型狀態
- `GET /api/classes` - 獲取所有類別名稱
- `POST /api/predict` - 圖片識別
- `POST /api/analyze-quality` - 圖片質量分析
- `POST /api/statistics` - 獲取統計數據
- `POST /api/chat` - AI聊天助手

## 🛠️ 開發環境

- Python 3.8+
- Node.js 16+
- TensorFlow 2.15.0+
- React 18.2.0
- Flask 3.0.0
- OpenCV 4.8.0+

## ⚠️ 注意事項

1. **Git LFS**: 模型文件使用Git LFS存儲，克隆後需要運行 `git lfs install`
2. **首次運行**: 首次運行需要加載模型，可能需要一些時間
3. **GPU加速**: 訓練模型建議使用GPU加速（Google Colab推薦）
4. **磁盤空間**: 確保有足夠的磁盤空間存儲數據集和模型（模型約19MB）
5. **瀏覽器兼容性**: 建議使用Chrome、Firefox或Edge最新版本

## 📝 更新日誌

### v1.0.0 (最新)
- ✨ 新增收藏功能
- ✨ 新增圖片質量分析
- ✨ 新增AI聊天助手
- ✨ 新增識別歷史統計和分析
- ✨ 新增批量識別模式
- 🐛 修復分類分布問題（蝴蝶正確分類）
- 📦 使用Git LFS管理大文件

## 📚 參考資料

- [TensorFlow官方文檔](https://www.tensorflow.org/)
- [Keras遷移學習指南](https://keras.io/guides/transfer_learning/)
- [React官方文檔](https://react.dev/)
- [Flask官方文檔](https://flask.palletsprojects.com/)
- [Git LFS文檔](https://git-lfs.github.com/)

## 📄 授權

本項目僅用於學術和教育目的。

## 👥 貢獻

歡迎提交Issue和Pull Request！

## 📧 聯繫方式

如有問題或建議，請通過GitHub Issues聯繫。

---

**注意**: 請確保在提交前完成所有必要的配置和測試。

## 📄 其他語言版本

- [English Version](README.md)
- [简体中文版 (Simplified Chinese)](README.zh-CN.md)

