## Movie Recommendor

A recommendation engine filters the data using different algorithms and recommends the most relevant items to users. It first captures the past behavior of a customer and based on that, recommends products which the users might be likely to buy on Amazon, watch on Netflix or search on Google.

In this article, we will cover various types of recommendation engine algorithms and fundamentals of creating them in Python. We will also see the mathematics behind the workings of these algorithms. Finally, we will create our own recommendation engine

**Collecting Data**:

This is the first and most crucial step for building a recommendation engine. The data can be collected by two means: explicitly and implicitly. Explicit data is information that is provided intentionally, i.e. input from the users such as movie ratings. Implicit data is information that is not provided intentionally but gathered from available data streams like search history, clicks, order history, etc.

For our usecase here we will use the pre collected dataset of Movielens.

> Code snippet from data_loader.py
```
class DataLoaderExplorer():
    """
    users: Table of users
    users_ratings: Table of users with ratings count and ratings mean
    movies: Table of movies
    movies_ratings: Table of movies with ratings count and ratings mean
    # Ratings Count: Total number of ratings given by user or given to movies
    # Ratings Mean: Mean rating given by user or given to movies
    """
    def __init__(self):
        self.users = None
        self.movies = None
        self.ratings = None
        self.holdout_fraction = 0.1
        self.users_ratings = None
        self.movies_ratings = None
        self.occupation_chart = None
        self.occupation_filter = None
        self.age_chart = None
        self.age_filter = None
        self.genre_filter = None
        self.genre_chart = None
        self.genre_cols = []
        self.users_cols = []
        self.movies_cols = []
        self.ratings_cols = []
        self.df = None
        self.train_df = None
        self.test_df = None
        
        def mark_genres(self):
        ...
        
        def load_movielens_dataset(self):
        ...
```

**Exploring Dataset**:

**Filtering Data**:
* Content based filtering
* User-User collaborative filtering
* Item-Item collaborative filtering

**Similarities Techniques**:
* Cosine
* Dot Product
* Euclidean Distance
* Pearson's Correlation

**Demo**:
[Link to Google Collab](https://colab.research.google.com/drive/1NiZueMUlvaUzn0yzuIZOZPT3hC4gLRws)
