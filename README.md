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

