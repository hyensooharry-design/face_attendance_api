import os
import faiss
import numpy as np

# Create folder if it doesn't exist
os.makedirs("faiss_index", exist_ok=True)

# Example: Generate fake embeddings and names
# Replace this with actual embeddings from face images
num_faces = 5
embedding_dim = 512
embeddings = np.random.rand(num_faces, embedding_dim).astype('float32')
names = np.array(["Alice", "Bob", "Charlie", "David", "Eva"])

# Create FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Save
faiss.write_index(index, "faiss_index/face.index")
np.save("faiss_index/names.npy", names)

print("FAISS index and names.npy created successfully!")
