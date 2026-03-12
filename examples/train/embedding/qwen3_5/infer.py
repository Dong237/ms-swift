# Qwen3.5 Embedding Inference Example
# For full-parameter training, use the checkpoint path directly.
# For LoRA training, pass the adapter path via `adapters` parameter.

import torch

from swift.infer_engine import InferRequest, TransformersEngine


def run_qwen3_5_emb():
    # For full-parameter trained model, use the checkpoint path directly:
    engine = TransformersEngine(
        'output/qwen3_5_emb_phase1/checkpoint-xxx',  # Replace with your checkpoint path
        model_type='qwen3_5_emb',
        task_type='embedding',
        torch_dtype=torch.bfloat16)

    # For LoRA fine-tuned model, use:
    # engine = TransformersEngine(
    #     'Qwen/Qwen3.5-0.8B',
    #     model_type='qwen3_5_emb',
    #     task_type='embedding',
    #     torch_dtype=torch.bfloat16,
    #     adapters=['output/vx-xxx/checkpoint-xxx'])

    infer_requests = [
        InferRequest(messages=[{'role': 'user', 'content': 'What is the capital of China?'}]),
        InferRequest(messages=[{'role': 'user', 'content': 'The capital of China is Beijing.'}]),
        InferRequest(messages=[{'role': 'user', 'content': 'A cat is sleeping on the sofa.'}]),
    ]
    resp_list = engine.infer(infer_requests)
    embeddings = []
    for resp in resp_list:
        embeddings.append(torch.tensor(resp.data[0].embedding))
    embedding_matrix = torch.stack(embeddings)
    # Compute cosine similarity matrix
    norms = embedding_matrix.norm(dim=1, keepdim=True)
    similarity = (embedding_matrix @ embedding_matrix.T) / (norms @ norms.T)
    print(f'Cosine similarity matrix:\n{similarity}')


if __name__ == '__main__':
    run_qwen3_5_emb()
