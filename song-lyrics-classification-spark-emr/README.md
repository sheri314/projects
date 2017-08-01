# Song Lyric Exploration via Spark / Amazon EMR

[Presentation](https://github.com/CameronSCarlin/MSAN624_Project/blob/master/Song%20Clustering%20Presentation.pdf)

[Kaggle Data Source](https://www.kaggle.com/mousehead/songlyrics)

## Data Information
- 57,650 songs
- Contains Artist, Song Name, Link, and Lyrics
- Lyrics were scraped from LyricsFreak

## Project Description
* Data Cleaning (Pandas):
  - removed URL column
  - removed punctuation
  - lower and remove punctuation titles, artists, and lyrics

* Aggregations (RDD/DataFrame/SQL):
  - most common artists
  - most common words by artist
  - most common words via WordClouds
  - feature creation: number of words, average word length, same for song title
  
* Machine Learning (Spark ML):
  - word2vec for numeric representations of songs
  - cluster by similarity, iterate through cluster sizes
