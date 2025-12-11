"""
推理模塊：生成解讀文本
用於加載訓練好的模型並生成物種解讀文本
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

class InterpretationGenerator:
    """
    解讀文本生成器
    """
    def __init__(self, model_path='models/interpretation_model', device='cpu'):
        """
        初始化生成器
        
        Args:
            model_path: 訓練好的模型路徑
            device: 運行設備
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load_model(self):
        """
        加載訓練好的模型
        """
        if self.loaded:
            return
        
        if not os.path.exists(self.model_path):
            print(f"⚠️ 模型路徑不存在: {self.model_path}")
            print("💡 請先訓練模型或檢查模型路徑")
            return False
        
        try:
            print(f"📥 正在加載模型: {self.model_path}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"✅ 模型加載成功 (設備: {self.device})")
            return True
        except Exception as e:
            print(f"❌ 模型加載失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(
        self,
        species,
        category,
        max_length=256,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    ):
        """
        Generate interpretation text
        
        Args:
            species: Species name (e.g., "臺灣藍鵲" - Chinese name)
            category: Category (e.g., "fun_fact", "behavior", "habitat")
            max_length: Maximum generation length
            temperature: Temperature parameter (controls randomness, higher = more random)
            top_p: Nucleus sampling parameter (controls diversity)
            num_return_sequences: Number of sequences to generate
        
        Returns:
            generated_text: Generated interpretation text in English, None if failed
        """
        if not self.loaded:
            if not self.load_model():
                return None
        
        try:
            # Build input text (species name in Chinese, but output will be in English)
            input_text = f"{species}的{category}："
            
            # Encode input
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            
            # Generate text
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2  # Repetition penalty to avoid repetitive generation
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Remove input part, keep only generated part
            generated_text = generated_text.replace(input_text, "").strip()
            
            # Clean text (remove special characters and extra spaces)
            generated_text = generated_text.replace("\n", " ").strip()
            
            return generated_text
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# 全局生成器實例（用於單例模式）
_global_generator = None

def get_generator(model_path='models/interpretation_model', device='cpu'):
    """
    獲取全局生成器實例（單例模式）
    
    Args:
        model_path: 模型路徑
        device: 設備
    
    Returns:
        generator: 生成器實例
    """
    global _global_generator
    
    if _global_generator is None:
        _global_generator = InterpretationGenerator(model_path, device)
        _global_generator.load_model()
    
    return _global_generator

if __name__ == "__main__":
    # 測試生成
    print("=" * 60)
    print("🧪 測試解讀文本生成")
    print("=" * 60)
    
    # 創建生成器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = InterpretationGenerator(
        model_path='models/interpretation_model',
        device=device
    )
    
    # 測試生成臺灣藍鵲的fun_fact
    print("\n📝 測試生成：臺灣藍鵲的fun_fact")
    print("-" * 60)
    
    result = generator.generate(
        species="臺灣藍鵲",
        category="fun_fact",
        max_length=256,
        temperature=0.7
    )
    
    if result:
        print(f"✅ 生成成功：")
        print(f"{result}")
    else:
        print("❌ 生成失敗（可能是模型未訓練或路徑錯誤）")
        print("💡 提示：請先運行 train_interpretation_model.py 訓練模型")

