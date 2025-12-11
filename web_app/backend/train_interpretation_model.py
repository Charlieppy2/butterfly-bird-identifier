"""
訓練腳本：結構化解讀生成模型
用於訓練一個能夠根據物種和類別生成解讀文本的模型

訓練流程：
1. 加載和預處理數據
2. 準備數據集和數據加載器
3. 設置訓練參數
4. 訓練循環（前向傳播、損失計算、反向傳播）
5. 評估（困惑度、BLEU分數）
6. 保存模型
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

# 設置設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

class InterpretationDataset(Dataset):
    """
    數據集類
    用於加載和處理訓練數據
    """
    def __init__(self, csv_path, tokenizer, max_length=256):
        """
        初始化數據集
        
        Args:
            csv_path: CSV文件路徑
            tokenizer: 分詞器
            max_length: 最大序列長度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 讀取CSV數據
        df = pd.read_csv(csv_path)
        self.data = df.to_dict('records')
        
        print(f"✅ 加載了 {len(self.data)} 條訓練數據")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        獲取一條數據
        
        Returns:
            input_ids: 輸入ID
            attention_mask: 注意力掩碼
            labels: 標籤（用於計算損失）
        """
        item = self.data[idx]
        
        # 構建輸入：物種 + 類別
        species = item['species']
        category = item['category']
        interpretation = item['interpretation_text']
        
        # Input format: "{species}的{category}：{interpretation}"
        # Note: species name is in Chinese, but interpretation text is in English
        input_text = f"{species}的{category}：{interpretation}"
        
        # 使用分詞器編碼
        encoded = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # 標籤就是input_ids（用於計算損失）
        labels = input_ids.clone()
        
        # 將padding位置的標籤設為-100（損失計算時會忽略）
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def calculate_perplexity(model, dataloader, device):
    """
    計算困惑度（Perplexity）
    困惑度越低，模型越好
    
    Args:
        model: 模型
        dataloader: 數據加載器
        device: 設備
    
    Returns:
        perplexity: 困惑度值
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="計算困惑度"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 計算有效token數量（排除padding）
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def calculate_bleu(model, tokenizer, test_data, device, num_samples=10):
    """
    計算BLEU分數
    BLEU分數越高，生成文本質量越好
    
    Args:
        model: 模型
        tokenizer: 分詞器
        test_data: 測試數據
        device: 設備
        num_samples: 計算BLEU的樣本數量
    
    Returns:
        avg_bleu: 平均BLEU分數
    """
    model.eval()
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    # 隨機選擇樣本進行評估
    sample_indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)
    
    with torch.no_grad():
        for idx in sample_indices:
            item = test_data[idx]
            species = item['species']
            category = item['category']
            reference = item['interpretation_text']
            
            # 生成文本
            input_text = f"{species}的{category}："
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            
            # 生成
            output = model.generate(
                input_ids,
                max_length=256,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 解碼生成的文本
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # 移除輸入部分，只保留生成部分
            generated_text = generated_text.replace(input_text, "").strip()
            
            # 計算BLEU分數
            reference_tokens = list(reference)
            generated_tokens = list(generated_text)
            
            try:
                bleu = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
                bleu_scores.append(bleu)
            except:
                pass
    
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    return avg_bleu

def train_model(
    csv_path='training_data_interpretation.csv',
    output_dir='models/interpretation_model',
    num_epochs=5,
    batch_size=4,
    learning_rate=5e-5,
    max_length=256,
    warmup_steps=100
):
    """
    訓練模型的主函數
    
    Args:
        csv_path: 訓練數據CSV路徑
        output_dir: 模型保存目錄
        num_epochs: 訓練輪數
        batch_size: 批次大小
        learning_rate: 學習率
        max_length: 最大序列長度
        warmup_steps: 預熱步數
    """
    print("=" * 60)
    print("🚀 開始訓練結構化解讀生成模型")
    print("=" * 60)
    
    # 1. 加載模型和分詞器
    print("\n📥 步驟1: 加載預訓練模型...")
    from model_setup import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(device=device)
    
    # 2. 準備數據集
    print("\n📊 步驟2: 準備數據集...")
    dataset = InterpretationDataset(csv_path, tokenizer, max_length=max_length)
    
    # 劃分訓練集和驗證集（80/20）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"   訓練集: {len(train_dataset)} 條")
    print(f"   驗證集: {len(val_dataset)} 條")
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. 設置優化器和學習率調度器
    print("\n⚙️ 步驟3: 設置優化器...")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"   總訓練步數: {total_steps}")
    print(f"   預熱步數: {warmup_steps}")
    
    # 4. 訓練循環
    print("\n🎯 步驟4: 開始訓練...")
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"📅 Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # 訓練階段
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"訓練 Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(progress_bar):
            # 將數據移到設備
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向傳播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            # 更新進度條
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 驗證階段
        print(f"\n🔍 驗證階段...")
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="驗證"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # 計算困惑度
        print(f"\n📊 計算困惑度...")
        perplexity = calculate_perplexity(model, val_loader, device)
        
        # 計算BLEU（每2個epoch計算一次，因為比較耗時）
        bleu_score = 0.0
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            print(f"📊 計算BLEU分數...")
            val_data = [dataset.data[i] for i in val_dataset.indices]
            bleu_score = calculate_bleu(model, tokenizer, val_data, device, num_samples=5)
        
        # 記錄訓練歷史
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'perplexity': perplexity,
            'bleu': bleu_score
        }
        training_history.append(epoch_history)
        
        print(f"\n📈 Epoch {epoch + 1} 結果:")
        print(f"   訓練損失: {avg_train_loss:.4f}")
        print(f"   驗證損失: {avg_val_loss:.4f}")
        print(f"   困惑度: {perplexity:.2f}")
        if bleu_score > 0:
            print(f"   BLEU分數: {bleu_score:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"\n💾 保存最佳模型 (驗證損失: {avg_val_loss:.4f})...")
            
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # 保存訓練歷史
            with open(os.path.join(output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
                json.dump(training_history, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ 訓練完成！")
    print(f"{'='*60}")
    print(f"📁 模型已保存到: {output_dir}")
    print(f"📊 最佳驗證損失: {best_val_loss:.4f}")

if __name__ == "__main__":
    # 訓練參數
    train_model(
        csv_path='training_data_interpretation.csv',
        output_dir='models/interpretation_model',
        num_epochs=5,
        batch_size=4,
        learning_rate=5e-5,
        max_length=256,
        warmup_steps=100
    )

