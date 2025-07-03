# Music Recommendation Engine

This project is a machine learning-based song recommendation engine that suggests similar songs based on audio features, artist, and genre using content-based filtering.

## Features
- Recommends songs similar to a user-selected track
- Uses features such as artist, genre, popularity, tempo, and valence
- Content-based filtering with scikit-learn

## How It Works
1. The system reads a dataset of songs and their features from `dataset.csv`, imported from Kaggle
2. It combines selected features into a single string for each song.
3. It uses `CountVectorizer` and cosine similarity to find songs most similar to the user's input.
4. The user enters a song they like, and the system outputs a list of similar songs.

## Getting Started

### Prerequisites
- Python 3.12 or higher
- pandas
- numpy
- scikit-learn

You can install the dependencies with:
```bash
pip install pandas numpy scikit-learn
```

### Usage
1. Place your `dataset.csv` file in the project directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Enter the name of a song when prompted to receive recommendations.

## Project Structure
- `main.py` — Main script for running the recommendation system
- `dataset.csv` — Dataset of songs and features
- `pyproject.toml` — Project metadata and dependencies

## License
This project is for educational purposes.
