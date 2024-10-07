from pymilvus import MilvusClient
import numpy as np
import os 

if os.path.exists("./milvus.db"):
    os.remove("./milvus.db")

client = MilvusClient("./milvus.db")
dim = 2048
client.create_collection(
    collection_name="collection",
    dimension=dim
)

actions = [
    'sit', 'stand', 'walk', 'run', 'jump', 'climb', 'swim', 'sit', 'drive', 'ride', 'sail', 'sit', 'sit', 'surf', 'dive', 'sit', 'hunt', 'shoot'
]

vectors = [np.random.random(dim) for _ in range(len(actions))]
data = [ {"id": i, "vector": vectors[i], "actions": actions[i]} for i in range(len(vectors)) ]
res = client.insert(
    collection_name="collection",
    data=data
)

res = client.search(
    collection_name="collection",
    data=[vectors[0]],
    limit=5
)
print(res[0])

# res = client.query(
#     collection_name="collection",
#     filter="actions == 'sit'",
# )
# print(res[0])

# res = client.delete(
#     collection_name="demo_collection",
#     filter="subject == 'history'",
# )
# print(res)
