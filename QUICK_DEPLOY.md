# 🚀 快速部署指南

## 最簡單的部署方法（5分鐘）

### 方法一：使用 Vercel + Railway（推薦 ⭐）

#### 1️⃣ 部署後端到 Railway（2分鐘）

1. 訪問 https://railway.app
2. 用 GitHub 登錄
3. 點擊 "New Project" → "Deploy from GitHub repo"
4. 選擇你的倉庫 `butterfly-bird-identifier`
5. 設置：
   - **Root Directory**: `web_app/backend`
   - **Start Command**: `python app.py`
6. 等待部署完成，複製你的後端 URL（例如：`https://xxx.railway.app`）

#### 2️⃣ 部署前端到 Vercel（3分鐘）

1. 訪問 https://vercel.com
2. 用 GitHub 登錄
3. 點擊 "New Project"
4. 選擇你的倉庫 `butterfly-bird-identifier`
5. 設置：
   - **Framework Preset**: Create React App
   - **Root Directory**: `web_app/frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
6. 添加環境變量：
   - **Name**: `REACT_APP_API_URL`
   - **Value**: 你剛才複製的 Railway URL
7. 點擊 "Deploy"

✅ 完成！你的網站已經上線了！

---

## 本地測試構建（部署前測試）

### Windows 用戶：

```bash
cd web_app
build_and_test.bat
```

### Mac/Linux 用戶：

```bash
cd web_app
chmod +x build_and_test.sh
./build_and_test.sh
```

構建完成後，測試生產版本：

```bash
# 安裝 serve（如果還沒安裝）
npm install -g serve

# 啟動測試服務器
cd web_app/frontend/build
serve -s . -l 3000
```

然後訪問 http://localhost:3000

---

## 常見問題

### ❓ 部署後無法連接後端？

1. 檢查 Vercel 環境變量 `REACT_APP_API_URL` 是否正確
2. 檢查 Railway 後端是否正在運行
3. 檢查瀏覽器控制台是否有 CORS 錯誤

### ❓ 圖片上傳失敗？

1. 檢查文件大小（最大 16MB）
2. 檢查後端日誌是否有錯誤

### ❓ 模型文件太大？

- 模型文件已經通過 Git LFS 管理
- Railway 會自動下載，無需手動操作

---

## 需要更多幫助？

查看詳細部署指南：`DEPLOYMENT.md`

---

## 部署檢查清單

- [ ] 後端已部署到 Railway
- [ ] 前端已部署到 Vercel
- [ ] 環境變量已正確設置
- [ ] 測試圖片上傳功能
- [ ] 測試識別功能
- [ ] 測試收藏功能
- [ ] 測試統計功能

完成所有檢查後，你的網站就準備好了！🎉

