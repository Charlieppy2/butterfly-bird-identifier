# 快速開始指南

## 前置要求

1. **Python 3.8+** - [下載地址](https://www.python.org/downloads/)
2. **Node.js 16+** - [下載地址](https://nodejs.org/)
3. **Git** (可選) - [下載地址](https://git-scm.com/)

## 安裝步驟

### 1. 準備數據集

將您的圖片數據按以下結構組織：

```
data/raw/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   └── ...
└── ...
```

### 2. 數據預處理

```bash
cd models/training
python prepare_data.py
```

這會將數據分割為訓練集、驗證集和測試集。

### 3. 訓練模型

```bash
# 仍在 models/training 目錄下
python train_model.py
```

**注意**: 訓練可能需要較長時間，建議使用GPU（可在Google Colab上運行）

### 4. 安裝後端依賴

```bash
cd ../../web_app/backend
pip install -r requirements.txt
```

### 5. 安裝前端依賴

```bash
cd ../frontend
npm install
```

## 運行應用

### 方法1: 使用批處理文件（Windows）

1. 雙擊 `web_app/backend/start_backend.bat` 啟動後端
2. 雙擊 `web_app/frontend/start_frontend.bat` 啟動前端

### 方法2: 使用命令行

**終端1 - 啟動後端**:
```bash
cd web_app/backend
python app.py
```

**終端2 - 啟動前端**:
```bash
cd web_app/frontend
npm start
```

### 訪問應用

- 前端: http://localhost:3000
- 後端API: http://localhost:5000

## 使用說明

1. **上傳圖片**: 點擊 "Choose File" 按鈕選擇圖片
2. **使用攝像頭**: 點擊 "Open Camera" 使用設備攝像頭拍攝
3. **識別**: 點擊 "🔍 Identify" 按鈕進行識別
4. **查看結果**: 系統會顯示識別結果和置信度

## 常見問題

### Q: 模型訓練時出現內存不足錯誤
A: 嘗試減小 `BATCH_SIZE` 或使用更小的圖片尺寸

### Q: 後端無法加載模型
A: 確保模型文件 `models/trained/model.h5` 存在，且已訓練完成

### Q: 前端無法連接到後端
A: 檢查後端是否運行在 http://localhost:5000，並確認 `web_app/frontend/.env` 文件中的API URL正確

### Q: 攝像頭無法使用
A: 確保瀏覽器有攝像頭權限，且使用HTTPS或localhost

## 下一步

- 查看 `README.md` 了解詳細文檔
- 查看 `report/report_template.md` 了解報告格式
- 根據需要調整模型參數和UI設計

