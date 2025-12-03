import sys
import os
import json
from typing import Optional, Tuple, Dict

# ===================================================================
# 1. 路径修复 (关键修改)
# ===================================================================
# 强制将 judge 目录插入到 sys.path 的最前面 (索引0)
# 这样 Python 会优先在这里寻找 qwen_235B.py，而不是当前目录
JUDGE_DIR = "/data/zhuldz/lunwen/judge"
if JUDGE_DIR not in sys.path:
    sys.path.insert(0, JUDGE_DIR)

try:
    from qwen_235B import CModelJudge
    print(f"✅ 成功从 {JUDGE_DIR} 导入 CModelJudge")
except ImportError as e:
    # 如果还是失败，定义 Mock 类防止 crash
    print(f"❌ 导入 CModelJudge 失败: {e}")
    class CModelJudge:
        def __init__(self, **kwargs): pass
        def judge(self, **kwargs): return {"valid": False, "error": f"Import Failed: {e}"}

# ===================================================================
# 2. 接口定义
# ===================================================================
def load_c_model(
    api_url: str = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1/chat/completions",
    auth_token: str = "msk-8a895e7fa53a8785f9cc4dc0364fae9064ccc540bbd419b5ba7cde8340ec2af8"
) -> CModelJudge:
    """加载C模型评判器"""
    try:
        judge = CModelJudge(api_url=api_url, auth_token=auth_token)
        return judge
    except Exception as e:
        print(f"C模型评判器加载失败：{str(e)}")
        raise

def _try_fix_json_error(error_msg: str) -> Optional[float]:
    """尝试修复简单的 JSON 格式错误"""
    try:
        if "原始JSON部分" not in error_msg: return None
        target = "原始JSON部分：" if "原始JSON部分：" in error_msg else "原始JSON部分:"
        raw_json = error_msg.split(target)[-1].strip()
            
        for i in range(5):
            candidate = raw_json if i == 0 else raw_json[:-i]
            if not candidate: continue
            try:
                data = json.loads(candidate)
                if "total_score" in data: return float(data["total_score"])
            except json.JSONDecodeError: continue
        return None
    except Exception:
        return None

def compute_reward(
    c_judge: CModelJudge,
    generated_code: str,
    canonical_solution: str,
    generated_prompt: str,
    b_model_input: str,
    user_prompt: str,  # <--- [关键修复] 必须添加这个参数
    model_c: str = "/model/qwen3-235b-a22b"
) -> Tuple[float, Dict]:
    """
    调用C模型计算奖励
    Returns: (score, full_result_dict)
    """
    try:
        judge_result = c_judge.judge(
            model_c=model_c,
            prompt=generated_prompt,
            generated_code=generated_code,
            code_ground_truth=canonical_solution,
            b_model_input=b_model_input,
            user_prompt=user_prompt  # <--- [关键修复] 传递给 judge 方法
        )

        if judge_result.get("valid", False):
            score = float(judge_result["result"].get("total_score", 0))
            return score, judge_result["result"]
        
        else:
            # 尝试自动修复
            err_msg = judge_result.get("error", "未知错误")
            fixed_score = _try_fix_json_error(err_msg)
            
            if fixed_score is not None:
                return fixed_score, {"total_score": fixed_score, "reason": "Auto-fixed JSON", "valid": True}

            print(f"评判无效，奖励设为0（错误：{err_msg[:100]}...）")
            return 0.0, {"error": err_msg, "valid": False}

    except Exception as e:
        print(f"计算奖励异常: {str(e)}")
        return 0.0, {"error": str(e), "valid": False}