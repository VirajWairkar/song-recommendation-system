from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import difflib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

songs_data = pd.read_csv("spotify_millsongdata.csv")

songs_data = songs_data.drop_duplicates(subset=["song", "artist"])

songs_data.reset_index(drop=True, inplace=True)

selected_features = ["artist", "song", "text"]

for feature in selected_features:
  songs_data[feature] = songs_data[feature].fillna("")

songs_data['song'] = songs_data['song'].str.lower()
songs_data["artist"] = songs_data["artist"].str.lower()
songs_data["text"] = songs_data["text"].str.lower()

combined_features = songs_data["artist"]+' '+songs_data["song"]+' '+songs_data["text"]

vectorizer = TfidfVectorizer(stop_words='english')

feature_vectors = vectorizer.fit_transform(combined_features)

list_of_all_songs = songs_data['song'].tolist()

class SongRequest(BaseModel):
  song: str

@app.get("/")
def root():
  return {"message":"Song Recommendation API"}

@app.post("/recommend")
def recommend(request: SongRequest):
    song_name = request.song.lower()

    find_close_match = difflib.get_close_matches(
       song_name, list_of_all_songs, n=1, cutoff=0.4
       )

    if not find_close_match:
        return {"error": "Song not found"}

    close_match = find_close_match[0]

    index_of_the_song = songs_data[songs_data.song == close_match].index[0]

    similarity_scores = cosine_similarity(
        feature_vectors[index_of_the_song].reshape(1, -1),
        feature_vectors
    )

    similarity_score = list(enumerate(similarity_scores[0]))

    sorted_similar_songs = sorted(
        similarity_score, key=lambda x: x[1], reverse=True
    )

    recommendations = []

    for i, song in enumerate(sorted_similar_songs[1:30]):
        index = song[0]
        title_from_index = songs_data['song'].values[index]
        recommendations.append({
           "song": songs_data['song'].values[index],
           "artist": songs_data['artist'].values[index]
        })

    return {
        "matched_song": close_match,
        "recommendations": recommendations
    }
