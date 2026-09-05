#!/usr/bin/env python3
"""验证 DeepSeek thinking mode 对 temperature/输出的影响"""
import json
from openai import OpenAI
 
key = open('/root/.deepseek_key').read().strip().split('=', 1)[1]
client = OpenAI(api_key=key, base_url='https://api.deepseek.com')
 
msgs = [{'role': 'user', 'content': 'You are in a room. Your task is to cool a tomato and place it in the fridge. The fridge 1 is open, tomato 1 is on countertop 1. Output exactly one action: Action: <command>'}]
 
def call(name, temperature=0.0, **extra):
    resp = client.chat.completions.create(
        model='deepseek-v4-flash',
        messages=msgs,
        max_tokens=3000,
        temperature=temperature,
        **extra,
    )
    c = resp.choices[0]
    content = c.message.content or ''
    reasoning = getattr(c.message, 'reasoning_content', None) or ''
    usage = getattr(resp, 'usage', None)
    return {
        'name': name,
        'content': content[:120],
        'has_reasoning': len(reasoning) > 0,
        'reasoning_len': len(reasoning),
        'total_tokens': usage.total_tokens if usage else None,
    }
 
results = []
# 1. thinking enabled (默认) + temp=0 —— 两次采样看是否一致
results.append(call('enabled_temp0_run1'))
results.append(call('enabled_temp0_run2'))
# 2. thinking disabled + temp=0 —— 两次采样
results.append(call('disabled_temp0_run1', extra_body={'thinking': {'type': 'disabled'}}))
results.append(call('disabled_temp0_run2', extra_body={'thinking': {'type': 'disabled'}}))
# 3. thinking disabled + temp=0.5 —— 两次采样
results.append(call('disabled_temp05_run1', extra_body={'thinking': {'type': 'disabled'}}, temperature=0.5))
results.append(call('disabled_temp05_run2', extra_body={'thinking': {'type': 'disabled'}}, temperature=0.5))
 
for r in results:
    print(json.dumps(r, ensure_ascii=False))
