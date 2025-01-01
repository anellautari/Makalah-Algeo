import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Membuat dokumen untuk pengetesan
documents = [
    "Institut Teknologi Bandung adalah multikampus yang berada di empat tempat",
    "Teknik Informatika adalah salah satu jurusan yang ada di Institut Teknologi Bandung",
    "Jurusan Teknik Informatika berada di kampus Ganesha dan Jatinangor"
]

# Langkah 1: Menghitung dan membuat matriks TF-IDF
# Fungsi untuk menghitung Term Frequency (TF)
def compute_tf(doc):
    tf = {}
    words = doc.split()
    total_words = len(words)
    for word in words:
        tf[word] = tf.get(word, 0) + 1
    for word in tf:
        tf[word] /= total_words     
    return tf

# Fungsi untuk menghitung Inverse Document Frequency (IDF)
def compute_idf(docs):
    import math
    idf = {}
    total_docs = len(docs)
    all_words = set(word for doc in docs for word in doc.split())
    for word in all_words:
        doc_count = sum(1 for doc in docs if word in doc.split())
        idf[word] = math.log((1 + total_docs) / (1 + doc_count)) + 1
    return idf

# Fungsi untuk menghitung TF-IDF
def compute_tf_idf(tf, idf):
    tf_idf = {}
    for word, tf_value in tf.items():
        tf_idf[word] = tf_value * idf.get(word, 0)  # Mengalikan TF dengan IDF
    return tf_idf

# Menghitung TF dan IDF untuk setiap dokumen
tfs = [compute_tf(doc) for doc in documents]
idf = compute_idf(documents)
tf_idf_matrix = [compute_tf_idf(tf, idf) for tf in tfs]

# Mengonversi matriks TF-IDF ke dalam DataFrame
all_words = sorted(set(word for tf_idf in tf_idf_matrix for word in tf_idf))
tf_idf_df = pd.DataFrame(
    [[tf_idf.get(word, 0) for word in all_words] for tf_idf in tf_idf_matrix],
    columns=all_words,
    index=["D1", "D2", "D3"]
)

print("Langkah 1: Matriks TF-IDF")
print(tf_idf_df, "\n")

# Langkah 2: Menerapkan SVD
svd = TruncatedSVD(n_components=3)
U_k = svd.fit_transform(tf_idf_df.values)       # Matriks U
Sigma_k = svd.singular_values_                  # Matriks Sigma
V_k = svd.components_                           # Matriks V^T

print("Langkah 2: Hasil SVD")
print("Matriks U :")
print(pd.DataFrame(U_k, index=tf_idf_df.index, columns=["Component 1", "Component 2", "Component 3"]), "\n")
print("Matriks Sigma:")
print(pd.DataFrame(Sigma_k, index=["Sigma 1", "Sigma 2", "Sigma 3"], columns=["Value"]), "\n")
print("Matriks V^T:")
print(pd.DataFrame(V_k, columns=all_words, index=["Component 1", "Component 2", "Component 3"]), "\n")

# Langkah 3: Menghitung cosine similarity dengan query
query = "Teknik Informatika di Ganesha"

# Fungsi untuk menghitung TF-IDF untuk query
def compute_query_tf_idf(query, idf):
    query_tf = compute_tf(query)
    query_tf_idf = compute_tf_idf(query_tf, idf)
    return np.array([query_tf_idf.get(word, 0) for word in all_words])

query_tfidf = compute_query_tf_idf(query, idf)

# Menghitung representasi query di ruang laten
query_latent_space = np.dot(query_tfidf, np.dot(V_k.T, np.linalg.pinv(np.diag(Sigma_k))))

# Menghitung representasi dokumen di ruang laten
document_latent_space = np.dot(U_k, np.diag(Sigma_k))

# Menghitung cosine similarity
cosine_similarities = cosine_similarity([query_latent_space], document_latent_space)

print("Hasil Q_lat dan D_lat")
print("Q_lat (Query di Ruang Laten):")
print(query_latent_space, "\n")
for i, doc_lat in enumerate(document_latent_space):
    print(f"D{i+1}_lat (Dokumen {i+1} di Ruang Laten):")
    print(doc_lat, "\n")

print("Langkah 3: Cosine Similarity")
similarity_df = pd.DataFrame(cosine_similarities, columns=["D1", "D2", "D3"], index=["Query"])
print(similarity_df)
