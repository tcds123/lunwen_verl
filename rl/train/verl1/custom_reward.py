import os
import sys
import json
import logging
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import traceback
import re

# ===================================================================
# 0. 路径修复
# ===================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ===================================================================
# 1. 导入接口
# ===================================================================
B_MODEL = None
B_TOKENIZER = None
C_JUDGE = None
MODELS_LOADED = False

try:
    from verl1.b_interface import load_b_model, generate_code_from_prompt
    from verl1.c_interface import load_c_model, compute_reward as compute_c_reward
except Exception as e:
    def load_b_model(**kwargs): raise ImportError(f"b_interface import failed: {e}")
    def load_c_model(**kwargs): raise ImportError(f"c_interface import failed: {e}")
    def generate_code_from_prompt(**kwargs): return ["Error: Import failed"]
    def compute_c_reward(**kwargs): return 0.0

# 全局配置
ROLLOUT_DATA_DIR = "./a_model_grpo_standard/rollouts"
ITERATION_LOG_DIR = "./a_model_grpo_standard/iterations"
K_CODE_GEN = 2
ROLLOUT_N = 5 
CURRENT_ITERATION = 0
_CONFIG_INITIALIZED = False
SAMPLE_COUNT = 0

# ===================================================================
# 2. C模型评判模板
# ===================================================================
C_MODEL_TEMPLATE = """你是专业代码评判模型（C模型），需要对生成代码和其对应的提示词进行评分。请严格按照以下规则评估：

【评估对象】
- 提示词（A模型输出）：{prompt}
- B模型输入（合并提示词）：{b_model_input}
- 生成代码（B模型输出）：{generated_code}
- 代码真值（参考标准）：{code_ground_truth}

【评估维度及权重】
1. 符合提示词度（30%）：生成代码是否严格遵循B模型输入的要求（功能、格式、约束等）
2. 代码质量（30%）：代码内容语法正确性、逻辑完整性、可读性（命名规范、注释等）
3. 功能一致性（40%）：与真值代码的核心功能是否一致（输入输出、处理逻辑）
4. 模型生成代码含有自然语言内容，在总分基础上扣两分。

【评分规则】
- 总分：-10 ~ +10（分数越高表示提示词效果越好，A模型应被奖励）
  - +7~+10：优秀（完全符合B模型输入，代码质量高，功能一致）
  - +3~+6：良好（基本符合B模型输入，少量问题，功能基本一致）
  - -2~+2：一般（部分符合B模型输入，有明显问题，功能有偏差）
  - -6~-3：较差（很少符合B模型输入，严重问题，功能偏差大）
  - -10~-7：极差（完全不符合B模型输入，无法运行，功能错误）
- 必须给出具体扣分/加分理由，禁止模糊评价

【输出格式】（严格按照JSON格式输出，键名不可修改）
{{
  "total_score": 具体分数（整数）,
  "match_prompt": 布尔值（true/false，是否符合B模型输入）,
  "score_details": {{
    "prompt_match_score": 维度1得分（-4~+4）,
    "code_quality_score": 维度2得分（-3~+3）,
    "function_consistency_score": 维度3得分（-3~+3）
  }},
  "reason": "具体评价理由（分点说明）"
}}"""

# ===================================================================
# 3. 核心清洗函数
# ===================================================================
def _clean_prompt_content(dirty_text: str) -> str:
    """从包含ICL的文本中提取原始问题"""
    if not isinstance(dirty_text, str): return str(dirty_text)
    start_marker = "Original prompt:"
    end_marker = "Correct code:"
    start_idx = dirty_text.find(start_marker)
    end_idx = dirty_text.find(end_marker)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return dirty_text[start_idx + len(start_marker):end_idx].strip()
    return dirty_text

# ===================================================================
# 4. 模型懒加载
# ===================================================================
def _ensure_models_loaded():
    global B_MODEL, B_TOKENIZER, C_JUDGE, MODELS_LOADED
    if MODELS_LOADED: return True
    try:
        if "CUDA_VISIBLE_DEVICES" in os.environ and not os.environ["CUDA_VISIBLE_DEVICES"]:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        if B_MODEL is None: B_MODEL, B_TOKENIZER = load_b_model()
        if C_JUDGE is None: C_JUDGE = load_c_model()
        if B_MODEL is not None and C_JUDGE is not None:
            MODELS_LOADED = True
            return True
        return False
    except Exception as e:
        print(f"❌ [CustomReward] Model Init Failed: {e}")
        traceback.print_exc()
        return False

def _initialize_globals_from_config(config):
    global _CONFIG_INITIALIZED, K_CODE_GEN, ROLLOUT_DATA_DIR, CURRENT_ITERATION
    if _CONFIG_INITIALIZED: return
    if config:
        CURRENT_ITERATION = config.get('current_iteration', 0) if isinstance(config, dict) else getattr(config, 'current_iteration', 0)
        if hasattr(config, 'iterative_rl'):
            K_CODE_GEN = config.iterative_rl.code_generation_count or 2
            ROLLOUT_DATA_DIR = config.iterative_rl.rollout_data_dir
        elif isinstance(config, dict) and 'iterative_rl' in config:
            K_CODE_GEN = config['iterative_rl'].get('code_generation_count', 2)
            ROLLOUT_DATA_DIR = config['iterative_rl'].get('rollout_data_dir', ROLLOUT_DATA_DIR)
    os.makedirs(ROLLOUT_DATA_DIR, exist_ok=True)
    _CONFIG_INITIALIZED = True

# ===================================================================
# 5. 核心奖励函数
# ===================================================================
def compute_custom_reward(**kwargs):
    global SAMPLE_COUNT
    _ensure_models_loaded()
    
    # --- 1. 提取 Prompts (A输入) ---
    prompts = kwargs.get('data_sources')
    if prompts is None: prompts = kwargs.get('prompts')
    if prompts is None: prompts = kwargs.get('input_ids')
    if prompts is None: prompts = kwargs.get('inputs')
    
    # --- 2. 提取 Responses (A输出) ---
    responses = kwargs.get('solution_strs')
    if responses is None: responses = kwargs.get('responses')
    if responses is None: responses = kwargs.get('predictions')
    
    # --- 3. 提取真值 ---
    ground_truths = None
    if 'canonical_solution' in kwargs: ground_truths = kwargs['canonical_solution']
    elif 'output' in kwargs: ground_truths = kwargs['output']
    elif 'ground_truth' in kwargs: ground_truths = kwargs['ground_truth']
    elif 'ground_truths' in kwargs: ground_truths = kwargs['ground_truths']
    elif 'references' in kwargs: ground_truths = kwargs['references']
    
    if ground_truths is None and 'reward_model' in kwargs:
        rm = kwargs['reward_model']
        if isinstance(rm, dict):
            ground_truths = rm.get('ground_truth', rm.get('output'))
        elif isinstance(rm, (list, np.ndarray)) and len(rm) > 0 and isinstance(rm[0], dict):
             ground_truths = [item.get('ground_truth', item.get('output')) for item in rm]

    # --- 4. 提取 raw_input (B输入) ---
    raw_inputs = kwargs.get('raw_input')
    if raw_inputs is None and prompts is not None:
        raw_inputs = [_clean_prompt_content(str(p)) for p in prompts]

    # 快速失败
    if prompts is None or responses is None:
        return {"reward_tensor": torch.zeros(1), "reward_extra_info": {}}

    config = kwargs.get('config')
    _initialize_globals_from_config(config)

    # 格式化
    if isinstance(prompts, (np.ndarray, torch.Tensor)): prompts = prompts.tolist()
    if isinstance(raw_inputs, (np.ndarray, torch.Tensor)): raw_inputs = raw_inputs.tolist()
    if isinstance(responses, (np.ndarray, torch.Tensor)): responses = responses.tolist()
    if ground_truths is not None and isinstance(ground_truths, (np.ndarray, torch.Tensor)):
        ground_truths = ground_truths.tolist()
    
    if ground_truths is None: ground_truths = [None] * len(prompts)
    if raw_inputs is None: raw_inputs = [""] * len(prompts)

    n = 1
    if len(prompts) > 0 and len(responses) > len(prompts):
        n = len(responses) // len(prompts)
        
    all_scores = []
    all_traces = [] 
    response_index = 0
    
    for i, original_prompt_blob in enumerate(prompts):
        gt = ground_truths[i] if i < len(ground_truths) else None
        gt_str = str(gt) if gt else "N/A (Truth Not Found)"
        
        # B 模型需要的清洗输入
        actual_user_query = raw_inputs[i] if i < len(raw_inputs) else str(original_prompt_blob)
        
        end_index = min(response_index + n, len(responses))
        system_prompts_from_policy = responses[response_index : end_index]
        response_index += n
        
        for system_prompt in system_prompts_from_policy:
            SAMPLE_COUNT += 1
            
            # ===================================================================
            # 【KEY FIX】: 正则清洗 <think> 标签
            # 允许模型思考，但在传给 Model B 之前把思考过程切掉
            # ===================================================================
            raw_system_prompt = str(system_prompt)
            # 移除 <think>...</think> 之间的所有内容（包括换行符）
            cleaned_system_prompt = re.sub(r'<think>.*?</think>', '', raw_system_prompt, flags=re.DOTALL).strip()
            
            # 如果清洗后为空（极端情况），回退到原始输出或给出默认提示，避免 B 模型报错
            if not cleaned_system_prompt:
                print(f"⚠️ [Warning] Prompt cleaned to empty! Fallback to raw.")
                # 这里可以选择回退，或者给定一个空指令让 C 模型去惩罚它
                # cleaned_system_prompt = raw_system_prompt 
                cleaned_system_prompt = "Generate python code based on the user request." # 兜底策略

            trace_entry = {
                "timestamp": datetime.now().isoformat(),
                "sample_id": SAMPLE_COUNT,
                "iteration": CURRENT_ITERATION,
                "a_model": {
                    "input": str(original_prompt_blob), 
                    "raw_output": raw_system_prompt,        # 记录含思维链的原始输出
                    "output": cleaned_system_prompt         # 记录实际生效的指令（为了兼容旧日志格式保留键名output）
                },
                "b_model": [], 
                "c_model": {} 
            }
            avg_score = 0.0
            full_c_input_log = "" 
            
            if MODELS_LOADED:
                try:
                    # 1. A -> B (使用清洗后的 cleaned_system_prompt)
                    codes_dict_list = _generate_codes(cleaned_system_prompt, str(actual_user_query))
                    
                    # 2. B -> C (返回 tuple: avg_score, scores, full_results)
                    avg_score, individual_scores, full_results = _evaluate_codes(codes_dict_list, gt_str, str(actual_user_query))
                    
                    b_logs = []
                    for k_idx, code_item in enumerate(codes_dict_list):
                        b_input = str(code_item.get('b_model_input', ''))
                        gen_code = str(code_item.get('code', ''))
                        
                        b_logs.append({
                            "input": b_input,
                            "output": gen_code
                        })
                        
                        # 只记录第一个样本的 Prompt 作为代表
                        if k_idx == 0:
                            full_c_input_log = C_MODEL_TEMPLATE.format(
                                prompt=str(cleaned_system_prompt), # 这里的 prompt 也是清洗后的
                                b_model_input=b_input,
                                generated_code=gen_code,
                                code_ground_truth=gt_str 
                            )

                    trace_entry["b_model"] = b_logs
                    trace_entry["c_model"] = {
                        "input": full_c_input_log if full_c_input_log else "Error: No code generated",
                        "output": full_results, 
                        "avg_score": float(avg_score)
                    }
                    
                except Exception as e:
                    print(f"❌ Pipeline Error: {e}")
                    traceback.print_exc()
                    trace_entry["b_model"] = [{"input": "Error", "output": f"Pipeline Error: {e}"}]
                    trace_entry["c_model"] = {"input": "Pipeline Error", "output": 0.0}
            else:
                trace_entry["b_model"] = [{"input": "Error", "output": "Models Not Loaded"}]
                trace_entry["c_model"] = {"input": "Models Not Loaded", "output": 0.0}
            
            all_scores.append(float(avg_score))
            try: all_traces.append(json.dumps(trace_entry, ensure_ascii=False))
            except: all_traces.append("{}")

    if len(all_scores) < len(responses):
        diff = len(responses) - len(all_scores)
        all_scores.extend([0.0] * diff)
        all_traces.extend(["{}"] * diff)
        
    return {
        "reward_tensor": torch.tensor(all_scores, dtype=torch.float32),
        "reward_extra_info": {"abc_trace": all_traces}
    }

# ===================================================================
# 6. 辅助函数
# ===================================================================
def _generate_codes(system_prompt: str, prompt: str) -> List[Dict]:
    codes = []
    if B_MODEL is None:
        return [{'code': "Error: B_MODEL is None", 'b_model_input': "", 'system_prompt': system_prompt}] * K_CODE_GEN

    try:
        sys_p = str(system_prompt) if system_prompt is not None else ""
        usr_p = str(prompt) if prompt is not None else ""
        generated_codes = generate_code_from_prompt(
            B_MODEL, B_TOKENIZER, generated_prompt=sys_p, original_prompt=usr_p, k=K_CODE_GEN
        )
    except Exception as e:
        print(f"Error in Model B: {e}")
        generated_codes = [f"Error: {e}"] * K_CODE_GEN
        
    for code in generated_codes:
        b_input = f"{system_prompt}\n{prompt}"
        codes.append({
            'code': code, 
            'b_model_input': b_input, 
            'system_prompt': system_prompt 
        })
    return codes
    
def _evaluate_codes(codes: List[Dict], ground_truth: str, original_prompt: str) -> tuple:
    scores = []
    full_results = []
    
    if C_JUDGE is None: return 0.0, [0.0]*len(codes), []
    if not codes: return 0.0, [], []
    
    gt_str = str(ground_truth) if ground_truth is not None else ""

    for code_item in codes:
        try:
            score_val, full_res = compute_c_reward(
                c_judge=C_JUDGE,
                generated_code=code_item['code'],
                canonical_solution=gt_str,
                generated_prompt=code_item.get('system_prompt', ''), 
                b_model_input=code_item['b_model_input']
            )
        except Exception as e: 
            print(f"Error in Model C: {e}")
            score_val = 0.0
            full_res = {"error": str(e)}
            
        scores.append(score_val)
        full_results.append(full_res)
    
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, scores, full_results