from openai import OpenAI
import numpy as np
import os

client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY")
)
# client = OpenAI(
#     base_url="https://api.siliconflow.cn/v1",
#     api_key=os.getenv("OPENAI_API_KEY")  # ❗不要写死
# )

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="BAAI/bge-m3",   # ✔ 你选的这个是对的
        input=text,
        encoding_format="float"  # ✔ 官方支持参数
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))