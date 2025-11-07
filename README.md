# 🎬 Movie Recommendation System (BiLSTM + TMDB API + Netflix UI)

This project builds a powerful **content-based movie recommendation system** using:

✅ BiLSTM-based deep semantic movie embeddings  
✅ Genre prediction as supervised signal  
✅ Cosine similarity  
✅ TMDB API (posters, cast, trailer, rating)  
✅ Netflix-style horizontal card UI (Jupyter widgets)

---

## ✅ Features
- Train a BiLSTM model on movie overviews  
- Generate 256-dim movie embeddings  
- Compute similarity matrix for recommendations  
- Recommend movies like “Avatar”, “Inception”, “Iron Man”, etc.  
- Fetch posters, cast, genres, trailers using TMDB API  
- Display recommendations in Netflix-style UI  

---

## ✅ Project Structure
├── recomandate_system2.ipynb # Main Notebook
├── recomandate_system2.py # Python script version
├── movie_titles.csv # Titles for lookup
├── dl_assets.pkl # Preprocessed tokenizer + mlb
├── tmdb_5000_movies.csv.zip # Dataset
├── tmdb_5000_credits.csv.zip # Dataset
├── README.md # Documentation

---

## ✅ Dataset Used
We use the **TMDB 5000 Movies Dataset**, which contains:

- title  
- overview  
- genres  
- cast & crew  
- movie_id  

---

## ✅ Model Architecture


Input (300 tokens)
↓
Embedding (300 × 128)
↓
BiLSTM (return_sequences=True → 300 × 256)
↓
Global Max Pooling → (256)
↓
Dense(256, relu) → Movie Embedding ✅
↓
Dropout(0.3)
↓
Dense(num_genres, sigmoid) → Genre Prediction


✅ The **movie embedding (256-dim)** is used for similarity  
✅ Sigmoid layer predicts multi-label genres  


✅ Sigmoid layer kya predict karta hai?

Sigmoid har genre ke liye ek probability (0 se 1 ke beech) predict karta hai.

Matlab:

Agar 20 genres hain,

To sigmoid 20 numbers predict karega.

Example output:

[0.91, 0.83, 0.03, 0.12, 0.01, ..., 0.45]


Yeh values probabilities hoti hain.

✅ 1 movie → 20 sigmoid outputs (multi-label)

Agar tumhare dataset me:

✅ 20 genres hain
✔ Action
✔ Comedy
✔ Sci-Fi
✔ Drama
... etc.

To output layer:

Dense(20, activation='sigmoid')


Means har movie ke liye model output karta hai:

Genre	Sigmoid Output
Action	0.91
Adventure	0.83
Sci-Fi	0.76
Drama	0.12
Romance	0.01
✅ Sigmoid kya batata hai?
✅ 0.91 → 91% chance movie Action hai
✅ 0.83 → 83% chance movie Adventure hai
✅ 0.76 → 76% chance Sci-Fi hai


---

## ✅ Training

Loss: `binary_crossentropy`  
Optimizer: `Adam`  
Callbacks:  
- EarlyStopping  
- ModelCheckpoint  

model.fit(
X_train, Y_train,
validation_data=(X_val, Y_val),
epochs=8,
batch_size=64,
callbacks=[EarlyStopping, Checkpoint]
)


---

## ✅ Movie Embedding Extraction



embed_model = Model(model.input, model.get_layer("movie_embedding").output)
movie_vecs = embed_model.predict(X)
movie_vecs_norm = movie_vecs / ||movie_vecs||


---

## ✅ Cosine Similarity


Shape: `(num_movies, num_movies)`


✅ 2) Calculate ALL movie-to-movie similarity
similarity = cosine_similarity(movie_vecs_norm)


Yaha:

movie_vecs_norm = shape (N, 256)
→ N movies, 256-d embeddings

cosine_similarity(matrix) ka matlab:

har movie vector ko baaki sab movies ke vectors se compare karo

ek N × N similarity matrix banao

Example:

Agar 4800 movies hain:

Output shape:

(4800, 4800)





---

## ✅ Recommendation Function

def recommend(movie, k=5):
idx = title_to_idx[movie]
sims = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
sims = sims[1:k+1] # skip itself
return [titles[i] for i,_ in sims]


---

## ✅ Netflix-Style UI (with Posters, Cast, Trailer)

Uses TMDB API:


https://api.themoviedb.org/3/movie/{id}?api_key=API_KEY&append_to_response=credits,videos


UI includes:

✅ Poster  
✅ Genre chips  
✅ Rating  
✅ Cast images  
✅ Year  
✅ Watch trailer button  
✅ Horizontal scroll cards  

---

## ✅ Requirements


pandas
numpy
scikit-learn
tensorflow
tqdm
requests
ipywidgets

TMDB_API_KEY = "YOUR_API_KEY"


---

## ✅ How to Run
1. Clone repository  
2. Extract TMDB datasets  
3. Open notebook `recomandate_system2.ipynb`  
4. Run all cells  
5. Choose a movie from dropdown  
6. Click **Show Recommendations**  
7. Enjoy Netflix-style output  

---

## ✅ Future Improvements
- Attention-based BiLSTM  
- Add cast + keyword embeddings  
- FAISS for instant similarity search  
- Streamlit Web App version  
- Combine collaborative + content-based methods  

---

## ✅ Author
**Kuldeep Kumar (k953)**  
Deep Learning • NLP • Recommender Systems

