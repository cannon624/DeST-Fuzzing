import json

import requests
# torch, fastchat, vllm are only needed for LocalLLM/LocalVLLM; make them optional
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from fastchat.model import load_model, get_conversation_template
    _HAS_FASTCHAT = True
except ImportError:
    _HAS_FASTCHAT = False

try:
    from vllm import LLM as vllm
    from vllm import SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

from openai import OpenAI
import logging
import time
import concurrent.futures
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class APIBasedLLM(LLM):
    """
    统一处理基于API的LLM的基类
    """
    def __init__(self, 
                 model_path, 
                 api_key, 
                 base_url,
                 system_message=None):
        super().__init__()
        
        if not api_key:
            raise ValueError('API key is required')
            
        self.model_path = model_path
        self.api_key = api_key
        self.base_url = base_url.rstrip('/') if base_url else None
        self.system_message = system_message
        
    def _make_request(self, payload, endpoint="/chat/completions", method="POST", timeout=60):
        """
        统一的API请求方法
        """
        try:
            headers = self._get_headers()
            
            url = f"{self.base_url}{endpoint}" if self.base_url else f"https://ent.openai.com/v1{endpoint}"
            
            if method.upper() == "POST":
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            else:
                response = requests.get(url, headers=headers, params=payload, timeout=timeout)
                
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logging.warning(f"API request failed: {e}")
            raise
            
    def _get_headers(self):
        """
        获取请求头
        """
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
    def _handle_response(self, data, n=1):
        """
        统一处理API响应
        """
        results = []
        try:
            for choice in data.get('choices', []):
                if 'message' in choice and 'content' in choice['message']:
                    results.append(choice['message']['content'])
        except Exception as e:
            logging.warning(f"Error parsing response: {e}")
            
        while len(results) < n:
            results.append(" ")
            
        return results[:n] if results else [" " for _ in range(n)]
        
    def _retry_request(self, request_func, max_trials=1, failure_sleep_time=1, *args, **kwargs):
        """
        统一的重试机制
        """
        for trial in range(max_trials):
            try:
                return request_func(*args, **kwargs)
            except Exception as e:
                logging.warning(
                    f"API call failed due to {e}. Retrying {trial+1} / {max_trials} times..."
                )
                if trial < max_trials - 1:  
                    time.sleep(failure_sleep_time)
        return None


class APILLM(APIBasedLLM):
    def __init__(self, 
                 model_path, 
                 api_key, 
                 base_url="https://ent.zetatechs.com/v1",
                 system_message=None,
                 provider="openai"):
        super().__init__(model_path, api_key, base_url, system_message)
        self.provider = provider.lower()
        
    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=3, failure_sleep_time=5):
        def _make_api_call():
            messages = self._build_messages(prompt)
            
            payload = self._build_payload(messages, temperature, max_tokens, n)
            
            data = self._make_request(payload)
            
            return self._handle_response(data, n)
            
        result = self._retry_request(_make_api_call, max_trials, failure_sleep_time)
        return result if result is not None else [" " for _ in range(n)]
        
    def _build_messages(self, prompt):
        messages = []
        
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
            
        if self.provider == "claude":
            formatted_prompt = f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"
            messages.append({
                "role": "user", 
                "content": formatted_prompt
            })
        else:
            messages.append({
                "role": "user",
                "content": prompt
            })
            
        return messages
        
    def _build_payload(self, messages, temperature, max_tokens, n):
        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n
        }
        
        if self.provider == "claude":
            payload["max_tokens"] = max_tokens
            payload.pop("n", None)
            
        return payload

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=3, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.generate, prompt, temperature, max_tokens, n, max_trials, failure_sleep_time): prompt 
                for prompt in prompts
            }
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results


class LocalLLM(LLM):
    def __init__(self,
                 model_path,
                 device='cuda',
                 num_gpus=4,
                 max_gpu_memory=None,
                 dtype=None,  # torch.float16
                 load_8bit=False,
                 cpu_offloading=False,
                 revision=None,
                 debug=False,
                 system_message=None
                 ):
        super().__init__()
        if not _HAS_TORCH or not _HAS_FASTCHAT:
            raise ImportError("torch and fastchat are required for LocalLLM. Install with: pip install torch fschat")
        if dtype is None:
            dtype = torch.float16

        self.model, self.tokenizer = self.create_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        self.model_path = model_path

        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    @torch.inference_mode()
    def create_model(self, model_path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=None,
                     dtype=torch.float16,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
        model, tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )

        return model, tokenizer

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    @torch.inference_mode()
    def generate(self, prompt, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        conv_temp = get_conversation_template(self.model_path)
        self.set_system_message(conv_temp)

        conv_temp.append_message(conv_temp.roles[0], prompt)
        conv_temp.append_message(conv_temp.roles[1], None)

        prompt_input = conv_temp.get_prompt()
        input_ids = self.tokenizer([prompt_input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

    @torch.inference_mode()
    def generate_batch(self, prompts, temperature=0.01, max_tokens=512, repetition_penalty=1.0, batch_size=16):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(prompt_inputs, padding=True).input_ids
        # load the input_ids batch by batch to avoid OOM
        outputs = []
        for i in range(0, len(input_ids), batch_size):
            output_ids = self.model.generate(
                torch.as_tensor(input_ids[i:i+batch_size]).cuda(),
                do_sample=False,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
        return outputs


class LocalVLLM(LLM):
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.95,
                 tensor_parallel_size=1, 
                 system_message=None
                 ):
        super().__init__()
        if not _HAS_VLLM:
            raise ImportError("vllm is required for LocalVLLM. Install with: pip install vllm")
        self.model_path = model_path
        self.model = vllm(
            self.model_path, 
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size) 
        
        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=0, max_tokens=512):
        prompts = [prompt]
        return self.generate_batch(prompts, temperature, max_tokens)

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(
            prompt_inputs, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs


class BardLLM(LLM):
    def generate(self, prompt):
        return


class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None,
                 base_url="https://ent.zetatechs.com/v1" 
                ):
        super().__init__()
        
        self.model_path = model_path
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=2, failure_sleep_time=5):
        for trial in range(max_trials):
            try:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                payload = {
                    "model": self.model_path,
                    "messages": [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "n": n
                }
                
                url = f"{self.base_url}/chat/completions"
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    return [choice['message']['content'] for choice in data['choices']]
                else:
                    logging.warning(
                        f"OpenAI API request failed with status code {response.status_code}: {response.text}. "
                        f"Retrying {trial+1}/{max_trials} times..."
                    )
                    time.sleep(failure_sleep_time)
                    
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {trial+1}/{max_trials} times..."
                )
                time.sleep(failure_sleep_time)
        
        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=3, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.generate, prompt, temperature, max_tokens, n, max_trials, failure_sleep_time): prompt 
                for prompt in prompts
            }
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results