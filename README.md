✅ README.md — Movie Recommendation System (BiLSTM + TMDB API + Netflix UI)
🎬 Movie Recommendation System

Deep Learning + NLP + LSTM-based Semantic Embedding + Cosine Similarity + TMDB API + Netflix-Style UI

This project builds a powerful content-based movie recommender using:
✅ Movie overview text
✅ Deep semantic movie embeddings (BiLSTM)
✅ Genre prediction as supervised signal
✅ Cosine similarity
✅ TMDB API for poster, cast, trailer
✅ A Netflix-like interactive UI (Jupyter widgets)

🚀 Features

✅ Train a BiLSTM model on movie overviews
✅ Learn 256-dim movie embeddings
✅ Build similarity matrix for recommendations
✅ Recommend movies like “Avatar”, “Inception”, etc.
✅ Fetch posters, genres, cast, trailers using TMDB API
✅ Display output in Netflix-style slider UI
✅ Supports Top-K similar movie recommendations

📂 Project Structure
├── recomandate_system2.ipynb      # Main Notebook
├── recomandate_system2.py         # Python script version
├── movie_titles.csv               # Titles & indexing
├── dl_assets.pkl                  # Processed tokenizer & mlb assets
├── tmdb_5000_movies.csv.zip       # Dataset (Movies)
├── tmdb_5000_credits.csv.zip      # Dataset (Credits)
├── README.md                      # Documentation

📊 Dataset Used
✅ TMDB 5000 Movies & Credits

Contains:

Title

Overview

Genres

Cast & Crew

Movie ID

🔥 Model Architecture
Input (300 tokens)
       ↓
Word Embedding (300 × 128)
       ↓
BiLSTM (return_sequences=True) → (300 × 256)
       ↓
Global Max Pooling → (256)
       ↓
Dense(256, relu) → Movie Embedding ✅
       ↓
Dropout(0.3)
       ↓
Dense(num_genres, sigmoid) → Genre Prediction


✅ The 256-dim Movie Embedding is used to compute similarity
✅ Genre prediction is used as supervised training signal

🧠 Training

We use:

✅ binary_crossentropy (multi-label)
✅ Adam optimizer
✅ EarlyStopping
✅ ModelCheckpoint

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=8,
    batch_size=64,
    callbacks=[EarlyStopping, Checkpoint]
)

🔍 Movie Embedding Extraction
embed_model = Model(inputs=model.input,
                    outputs=model.get_layer('movie_embedding').output)

movie_vecs = embed_model.predict(X)
movie_vecs_norm = movie_vecs / ||movie_vecs||

🔗 Cosine Similarity Matrix
similarity = cosine_similarity(movie_vecs_norm)


✅ Shape = (num_movies, num_movies)
✅ Used for Top-K recommendations

🎯 Recommendation Function
def recommend(movie, k=5):
    idx = title_to_idx[movie]
    sims = sorted(enumerate(similarity[idx]), reverse=True)
    return top-K most similar movies

🍿 Netflix-Style UI (With Posters, Cast, Trailer)

We use TMDB API:

https://api.themoviedb.org/3/movie/{id}?api_key=API_KEY&append_to_response=credits,videos


✅ Fetches poster
✅ Genres
✅ Cast images
✅ Rating
✅ Trailer (YouTube)

Rendered using HTML + ipywidgets:

Combobox → Movie selection
Button → Show Recommendations
HTML → Netflix slider cards

🖼️ Sample Output (UI)

✅ Horizontal scrolling movie cards
✅ Posters (HD)
✅ Cast photos
✅ Rating
✅ Genres
✅ Trailer button

🔧 Requirements
pandas
numpy
scikit-learn
tensorflow
tqdm
requests
ipywidgets


Install:

pip install pandas numpy scikit-learn tensorflow tqdm requests ipywidgets


Enable widgets in Jupyter:

jupyter nbextension enable --py widgetsnbextension

🔑 TMDB API Setup

Replace with your API key:

TMDB_API_KEY = "YOUR_API_KEY"

✅ How to Run

Download the repo

Unzip TMDB datasets

Open recomandate_system2.ipynb

Run all cells

Choose a movie from dropdown

Click Show Recommendations

Enjoy Netflix-style results 🔥

✅ Future Improvements

✅ Attention-based model
✅ Add cast + keywords embeddings
✅ Combine with collaborative filtering
✅ FAISS ANN search for ultra-fast similarity
✅ Streamlit web app version
