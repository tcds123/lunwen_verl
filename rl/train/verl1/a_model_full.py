import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import List
import os
import numpy as np

class AModelFull:
    """
    全参微调的A模型：基于SystemPromptGenerator的系统提示词生成器
    移除所有LoRA/PEFT相关逻辑
    """
    
    def __init__(self, model_path: str, device_id: int = 0, example_file: str = "/data/zhuldz/lunwen/rl/train/verl1/sysprompt_icl.txt"):
        """初始化全参微调的A模型"""
        # 设置目标设备
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|end_of_solution|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        # 加载基础模型
        logging.info(f"加载全参基础模型到设备: {self.device}")
        
        # 注意：全参微调通常需要配合 FSDP 使用。
        # 如果是在单卡上直接运行这个类（非FSDP环境），4B模型全参可能显存吃紧。
        # 这里我们按标准方式加载。
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, # 建议全参微调使用 bf16
            device_map={"": self.device} if torch.cuda.is_available() else None,
            trust_remote_code=True,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # 确保模型在正确设备上
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        # 启用梯度检查点以节省内存 (全参微调必须项)
        self._enable_gradient_checkpointing()
        
        # 设置为训练模式
        self.model.train()
        
        # 加载示例文件
        self.example_content = self._load_example_file(example_file)
        
        # 样本计数器
        self.sample_counter = 0
        self.max_example_samples = 5
        
        logging.info("全参A模型加载完成")

    def _enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logging.info("A模型已启用梯度检查点")
        else:
            logging.warning("A模型不支持梯度检查点")

    def _load_example_file(self, example_file):
        """加载示例文件"""
        try:
            if os.path.exists(example_file):
                with open(example_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logging.warning(f"示例文件 {example_file} 不存在")
                return ""
        except Exception as e:
            logging.error(f"加载示例文件失败: {str(e)}")
            return ""

    def generate_prompts(self, original_prompt: str, canonical_solution: str, m: int = 2, max_tokens: int = 1024):
        """生成系统提示词（推理模式）"""
        self.model.eval()
        try:
            prompt_template = self._build_prompt_template(original_prompt, canonical_solution)
            
            inputs = self.tokenizer(
                prompt_template,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                return_token_type_ids=False
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_return_sequences=m,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_prompts = []
            for output in outputs:
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                if "【model_Output】" in full_text:
                    split_parts = full_text.split("【model_Output】")
                    if split_parts:
                        prompt = split_parts[-1].strip()
                    else:
                        prompt = full_text.strip()
                else:
                    prompt = full_text.strip()
                generated_prompts.append(prompt)
            
            return generated_prompts, inputs, outputs
            
        except Exception as e:
            logging.error(f"生成提示词错误: {str(e)}")
            raise e 
        finally:
            self.model.train()

    def _build_prompt_template(self, original_prompt, canonical_solution):
        # ... (保持原有的模板构建逻辑不变) ...
        # 为了简洁，这里省略具体字符串拼接代码，请直接复制原文件中的 _build_prompt_template 方法
        instruction = "I will provide you with some examples of generating system prompts. Please carefully study and understand the content and structure of these examples.\n\n"
        
        example_block = ""
        if self.example_content and self.sample_counter < self.max_example_samples:
            example_block = f"Examples:\n{self.example_content}\n\n"
        
        generation_instruction = (
            "Based on the examples above, generate an English system prompt for the following input (follow the same format as examples),"
            "IMPORTANT RULES:\n"
            "Output ONLY the final system prompt, with NO intermediate thinking, explanations, or reasoning.\n"
            "Do NOT include phrases like 'Let me think', 'First, I need to', or any similar thought process.\n"
            "It is not allowed to output any thinking and explanatory statements, only the generated system prompts:\n"
        )
        
        task_input = (
            f"【Input】\n"
            f"Original prompt: {original_prompt}\n"
            f"Correct code: {canonical_solution}\n"
            f"【model_Output】\n"
        )
        
        self.sample_counter += 1
        return f"{instruction}{example_block}{generation_instruction}{task_input}"

    def reset_sample_counter(self):
        self.sample_counter = 0

    def save_weights(self, output_dir):
        """保存全参权重"""
        try:
            # 全参保存直接调用 save_pretrained
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logging.info(f"全参权重已保存到: {output_dir}")
        except Exception as e:
            logging.error(f"保存权重失败: {str(e)}")

    def load_weights(self, checkpoint_path):
        """加载全参权重"""
        try:
            # 全参加载直接覆盖当前模型
            # 注意：在FSDP中通常由Trainer处理，这里可能是用于推理或非分布式环境的加载
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(self.device)
            self.model.train()
            logging.info(f"全参权重已从 {checkpoint_path} 加载")
        except Exception as e:
            logging.error(f"加载权重失败: {str(e)}")