import sys
import torch
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# =============================================
# 模型选择配置区域
# 请根据需要注释/取消注释相应的代码段
# =============================================

# 选项1: 使用本地B模型 (默认启用)
# 取消注释以下代码段来使用本地模型
#USE_LOCAL_MODEL = True

# 选项2: 使用API版本的B模型
# 取消注释以下代码段来使用API版本
USE_API_MODEL = True

# =============================================
# 本地B模型实现 (当前启用)
# =============================================
if 'USE_LOCAL_MODEL' in locals() and USE_LOCAL_MODEL:
    # 将B模型脚本（gen_code.py）所在目录添加到Python路径
    sys.path.append("/data/zhuldz/lunwen/generation")

    # 从gen_code.py中导入核心函数
    from gen_code import (
        load_model_and_tokenizer,
        generate_code as b_model_generate_code
    )

    def load_b_model(model_path="/data/zhuldz/lunwen/models/Qwen2.5-Coder-7B-Instruct", device_id=1):
        """加载B模型到指定GPU设备（本地版本）"""
        # 设置目标设备
        target_device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # 将模型移动到指定设备
        if torch.cuda.is_available():
            model = model.to(target_device)
        
        logging.info(f"本地B模型已加载到设备: {target_device}")
        return model, tokenizer

    def smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens=4000):
        """
        智能合并提示词，优先保证原始提示词的完整性
        如果tokens超限，优先截断生成提示词
        """
        # 首先检查原始提示词是否超限
        original_tokens = tokenizer.encode(original_prompt, add_special_tokens=False)
        
        if len(original_tokens) >= max_tokens:
            # 如果原始提示词本身就超限，需要截断原始提示词
            truncated_original = tokenizer.decode(original_tokens[:max_tokens-100], skip_special_tokens=True)
            return f"[Original (truncated)]{truncated_original}"
        
        # 计算剩余可用tokens
        remaining_tokens = max_tokens - len(original_tokens) - 100  # 预留100tokens给格式标记
        
        if remaining_tokens <= 0:
            # 如果剩余tokens不足，只保留原始提示词
            return f"[Original]{original_prompt}"
        
        # 编码生成提示词
        generated_tokens = tokenizer.encode(generated_prompt, add_special_tokens=False)
        
        if len(generated_tokens) <= remaining_tokens:
            # 如果生成提示词可以完整保留
            return f"[Generated]{generated_prompt}\n[Original]{original_prompt}"
        else:
            # 需要截断生成提示词
            truncated_generated = tokenizer.decode(generated_tokens[:remaining_tokens], skip_special_tokens=True)
            return f"[Generated (truncated)]{truncated_generated}\n[Original]{original_prompt}"

    def generate_code_from_prompt(model, tokenizer, generated_prompt, original_prompt=None, k=1, max_tokens=4000):
        """
        调用本地B模型生成k个代码样本
        支持智能合并提示词，优先保证原始提示词的完整性
        """
        codes = []
        
        # 如果提供了original_prompt，进行智能合并
        if original_prompt is not None:
            combined_prompt = smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens)
        else:
            # 如果没有原始提示词，直接使用生成提示词（但需要检查长度）
            tokens = tokenizer.encode(generated_prompt, add_special_tokens=False)
            if len(tokens) > max_tokens:
                # 截断生成提示词
                combined_prompt = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            else:
                combined_prompt = generated_prompt
        
        for _ in range(k):
            # 调用原gen_code.py中的generate_code函数
            code = b_model_generate_code(model, tokenizer, combined_prompt)
            codes.append(code.strip())
        
        return codes

# =============================================
# API版本B模型实现 (当前禁用)
# =============================================
elif 'USE_API_MODEL' in locals() and USE_API_MODEL:
    # 将API版本脚本所在目录添加到Python路径
    sys.path.append("/data/zhuldz/lunwen/generation")

    # 从b_model_api.py中导入API版本
    from b_model_api import load_b_model, generate_code_from_prompt, smart_combine_prompts
    
    # 注意：API版本不需要重新定义这些函数，直接使用导入的函数
    logging.info("使用API版本的B模型")

# =============================================
# 默认配置：如果没有选择任何模型，使用本地模型
# =============================================
else:
    # 默认使用本地模型
    sys.path.append("/data/zhuldz/lunwen/generation")
    from gen_code import (
        load_model_and_tokenizer,
        generate_code as b_model_generate_code
    )

    def load_b_model(model_path="/data/zhuldz/lunwen/models/Qwen2.5-Coder-7B-Instruct", device_id=1):
        """加载B模型到指定GPU设备（默认本地版本）"""
        target_device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        if torch.cuda.is_available():
            model = model.to(target_device)
        
        logging.info(f"默认本地B模型已加载到设备: {target_device}")
        return model, tokenizer

    def smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens=4000):
        """智能合并提示词（默认本地版本）"""
        original_tokens = tokenizer.encode(original_prompt, add_special_tokens=False)
        
        if len(original_tokens) >= max_tokens:
            truncated_original = tokenizer.decode(original_tokens[:max_tokens-100], skip_special_tokens=True)
            return f"[Original (truncated)]{truncated_original}"
        
        remaining_tokens = max_tokens - len(original_tokens) - 100
        
        if remaining_tokens <= 0:
            return f"[Original]{original_prompt}"
        
        generated_tokens = tokenizer.encode(generated_prompt, add_special_tokens=False)
        
        if len(generated_tokens) <= remaining_tokens:
            return f"[Generated]{generated_prompt}\n[Original]{original_prompt}"
        else:
            truncated_generated = tokenizer.decode(generated_tokens[:remaining_tokens], skip_special_tokens=True)
            return f"[Generated (truncated)]{truncated_generated}\n[Original]{original_prompt}"

    def generate_code_from_prompt(model, tokenizer, generated_prompt, original_prompt=None, k=1, max_tokens=4000):
        """调用本地B模型生成代码（默认版本）"""
        codes = []
        
        if original_prompt is not None:
            combined_prompt = smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens)
        else:
            tokens = tokenizer.encode(generated_prompt, add_special_tokens=False)
            if len(tokens) > max_tokens:
                combined_prompt = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            else:
                combined_prompt = generated_prompt
        
        for _ in range(k):
            code = b_model_generate_code(model, tokenizer, combined_prompt)
            codes.append(code.strip())
        
        return codes


if __name__ == "__main__":
    try:
        # 加载B模型（根据配置选择版本）
        print("正在加载B模型...")
        model, tokenizer = load_b_model()
        
        # 测试智能合并功能
        test_generated = "As an AI code generation expert, you are tasked with producing high-quality, production-ready code based on user requirements. your primary objective is to generate code that is not only functionally correct but also exemplifies excellence in readability, performance, robustness, and adherence to established software engineering best practices. you must have no netural language!\nAnswer:\n\nAs an AI code generation expert, you are tasked with producing high-quality, production-ready code based on user requirements. your primary objective is to generate code that is not only functionally correct but also exemplifies excellence in readability, performance, robustness, and adherence to established software engineering best practices. you must have no netural language! pay attention to the tokens: file,extension,dict,paths,ignore,splitext,os,path,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,splitext,os,split","As an AI code generation expert, you are tasked with producing high-quality, production-ready code based on user requirements. your primary objective is to generate code that is not only functionally correct but also exemplifies excellence in readability, performance, robustness, and adherence to established software engineering best practices. you must have no netural language! please pay attention to the tokes：function,parameters,returns,code\nOkay, let's see. The user provided an example of how to generate a system prompt based on their input. They want me to create a similar prompt for their specific input about grouping files by extension.\n\nFirst, I need to understand the original prompt. The task is to write a Python function that takes a list of file paths and groups them into a dictionary by their extensions. The keys are extensions without the leading dot, and the values are lists of file paths. Files without extensions are ignored.\n\nLooking at the correct code, the function uses os.path.splitext to split the file path into the base and extension. It then checks if the extension exists, removes the leading dot, and adds the file to the corresponding list in the dictionary.\n\nNow, the example outputs have specific tokens that need to be included in the system prompt. For instance, \"function\", \"parameters\", \"returns\", and \"code\". Also, the previous examples mention things like handling edge cases, ensuring correctness, and following best practices.\n\nIn the given input, the correct code already includes a docstring with parameters and returns. The system prompt needs to emphasize producing code that's functional, readable, efficient, and follows best practices. It should also mention handling edge cases, such as files without extensions, and ensuring the dictionary is correctly structured.\n\nI need to make sure the generated system prompt doesn't include any neutral language and strictly follows the structure of the examples. The key points are to focus on the function's purpose, parameters, return value, and the code's quality. Also, the token \"code\" must be present, along with others like \"function\", \"parameters\", \"returns\".\n\nSo, putting it all together, the system prompt should start with the role as an AI code expert, state the objective of generating high-quality code, mention the key aspects like functionality, readability, performance, and best practices. Then specify the elements to focus on: function definition, parameters, return value, and the code itself. Ensure there's no extra explanation and just the prompt as per the examples.\n【Output】\nAs an AI code generation expert, you are tasked with producing high-quality, production-ready code based on user requirements. your primary objective is to generate code that is not only functionally correct but also exemplifies excellence in readability, performance, robustness, and adherence to established software engineering best practices. you must have no netural language! please pay attention to the tokes：function,parameters,returns,code. the code should correctly handle file paths with and without extensions, ensure the dictionary keys are file extensions without leading dots, and return a properly structured dictionary as specified. the code must be concise, well-documented, and free of errors. the function should process the input list efficiently and handle edge cases such as empty inputs or invalid file paths. the code must adhere to python syntax standards and include necessary imports. the output should be a precise, direct implementation of the requested functionality without any unnecessary explanations or supplementary text preceding or following the code itself. the code must be enclosed in triple backticks and follow the exact structure of the correct code provided. the function name, parameters, and return type must match the requirements exactly. the code must be optimized for performance and maintainability. the code must correctly categorize files by their extensions, ignoring those without extensions. the code must be thoroughly tested and free of logical errors. the code must be written in pure python and avoid any external dependencies beyond standard library modules. the code must be self-contained and ready for immediate use. the code must be formatted according to pycodestyle guidelines. the code must include clear and concise documentation for the function. the code must be written in a way that ensures the dictionary is built correctly and efficiently. the code must handle all possible input scenarios as described in the problem statement. the code must be written with the intention of being maintainable and extensible in the future. the code must be written with the intention of being used in a real-world application. the code must be written with the intention of being reviewed by other developers. the code must be written with the intention of being understood by other developers. the code must be written with the intention of being correct and reliable. the code must be written with the intention of being efficient and performant. the code must be written with the intention of being safe and secure. the code must be written with the intention of being scalable and adaptable. the code must be written with the intention of being clean and elegant. the code must be written with the intention of being correct and reliable. the code must be written with the intention of being efficient and performant. the code must"  # 模拟长提示词
        test_original = "You are given a list of file paths. Your task is to write a Python function that organizes these files into a dictionary based on their extensions. The keys in the dictionary should be the file extensions (without the leading dot), and the values should be lists of file paths that share the same extension. If a file does not have an extension, it should be ignored."
        
        print("\n正在生成代码...")
        codes = generate_code_from_prompt(
            model, tokenizer, 
            generated_prompt=test_generated,
            original_prompt=test_original,
            k=1
        )
        
        # 显示当前使用的模型类型
        if 'USE_API_MODEL' in locals() and USE_API_MODEL:
            print("当前使用：API版本的B模型")
        else:
            print("当前使用：本地版本的B模型")
        
        # 详细输出生成的代码结果
        print(f"\n{'='*60}")
        print("模型生成的代码结果：")
        print(f"{'='*60}")
        print(f"生成的代码数量：{len(codes)}")
        
        for i, code in enumerate(codes, 1):
            print(f"\n--- 第 {i} 个代码样本 ---")
            print(f"代码长度：{len(code)} 字符")
            print(f"代码内容：")
            print("-" * 40)
            print(code)
            print("-" * 40)
            
            # 显示代码的前100个字符作为预览
            if len(code) > 100:
                print(f"代码预览（前100字符）：{code[:100]}...")
            else:
                print(f"完整代码：{code}")
        
        print(f"\n{'='*60}")
        print("代码生成完成！")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"测试出错：{str(e)}")
        import traceback