'''
Author: shaoting0730 510738319@qq.com
Date: 2026-03-23 09:21:35
LastEditors: shaoting0730 510738319@qq.com
LastEditTime: 2026-03-23 10:01:52
FilePath: /bunny/Users/zhoushaoting/Desktop/GitHub/other-learn/AI入门/入门中/其它/大模型学习历程/day1/test_ai.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from openai import OpenAI

client = OpenAI(api_key="xxxxxx")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "用一句话解释什么是大模型"}
    ]
)

print(response.choices[0].message.content)