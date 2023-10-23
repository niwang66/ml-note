```python
embed_layer = nn.Embedding(num_embeddings=100, embedding_dim=5) #字典大小100， 词向量维度5
fc_layer = nn.Linear(5, 6, bias=False)# 输入5维，输出6维
x = torch.randint(100, size=(4, 16))# 4个样本，每个样本16个特征/Token
print(x.shape) # 4, 16
x = embed_layer(x) # [N,T] -> [N,T,E]
print(x.shape) # 4, 16, 5
x2 = fc_layer(x) # dot([N,T,E], [E,E2]) --> [N,T,E2]
print(x2.shape) # 4, 16, 6

print(x2[0][:3])
```