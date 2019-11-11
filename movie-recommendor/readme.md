## Movie Recommendor

A recommendation engine filters the data using different algorithms and recommends the most relevant items to users. It first captures the past behavior of a customer and based on that, recommends products which the users might be likely to buy on Amazon, watch on Netflix or search on Google.

In this article, we will cover various types of recommendation engine algorithms and fundamentals of creating them in Python. We will also see the mathematics behind the workings of these algorithms. Finally, we will create our own recommendation engine

**Collecting Data**:

This is the first and most crucial step for building a recommendation engine. The data can be collected by two means: explicitly and implicitly. Explicit data is information that is provided intentionally, i.e. input from the users such as movie ratings. Implicit data is information that is not provided intentionally but gathered from available data streams like search history, clicks, order history, etc.

For our casestudy we will work on the MovieLens dataset and build a model to recommend movies to the end users. This data has been collected by the GroupLens Research Project at the University of Minnesota. The dataset can be downloaded from here. This dataset consists of:
* 100,000 ratings (1-5) from 943 users on 1682 movies
* Demographic information of the users (age, gender, occupation, etc.)


> Code snippet from data_loader.py
```
def load_movielens_dataset(self):
    print("Downloading movielens data...")
    from urllib.request import urlretrieve
    import zipfile

    urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", "movielens.zip")
    zip_ref = zipfile.ZipFile('movielens.zip', "r")
    zip_ref.extractall()
    print("Done. Dataset contains:")
    print(zip_ref.read('ml-100k/u.info'))

    # The movies file contains a binary feature for each genre.
    self.genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    # Loading users dataset
    self.users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    self.users = pd.read_csv(
        'ml-100k/u.user', sep='|', names=self.users_cols, encoding='latin-1')
    # Since the ids start at 1, we shift them to start at 0.
    self.users["user_id"] = self.users["user_id"].apply(lambda x: str(x-1))

    # Loading movies dataset
    self.movies_cols = [
        'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
    ] + self.genre_cols
    self.movies = pd.read_csv(
        'ml-100k/u.item', sep='|', names=self.movies_cols, encoding='latin-1')
    # Since the ids start at 1, we shift them to start at 0.
    self.movies["movie_id"] = self.movies["movie_id"].apply(lambda x: str(x-1))
    self.movies["year"] = self.movies['release_date'].apply(lambda x: str(x).split('-')[-1])

    # Loading ratings dataset
    self.ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    self.ratings = pd.read_csv(
        'ml-100k/u.data', sep='\t', names=self.ratings_cols, encoding='latin-1')
    # Since the ids start at 1, we shift them to start at 0.
    self.ratings["movie_id"] = self.ratings["movie_id"].apply(lambda x: str(x-1))
    self.ratings["user_id"] = self.ratings["user_id"].apply(lambda x: str(x-1))
    self.ratings["rating"] = self.ratings["rating"].apply(lambda x: float(x))

    # Compute the number of movies to which a genre is assigned.
    genre_occurences = self.movies[self.genre_cols].sum().to_dict()

    # Create one merged DataFrame containing all the movielens data.
    movielens = self.ratings.merge(self.movies, on='movie_id').merge(self.users, on='user_id')
    self.df = movielens
    return movielens
```

**Exploring Dataset**:

Before we dive into model building, let's inspect our MovieLens dataset. It is usually helpful to understand the statistics of the dataset.

We start by printing the users dataset and creating some histograms associated to it.

> Code snippert from data_loader.py
```
def explore_user_data(self):
    self.users_ratings = (
        self.ratings
        .groupby('user_id', as_index=False)
        .agg({'rating': ['count', 'mean']})
        .flatten_cols()
        .merge(self.users, on='user_id')
    )

    self.occupation_filter = alt.selection_multi(fields=["occupation"])
    self.occupation_chart = alt.Chart().mark_bar().encode(
        x="count()",
        y=alt.Y("occupation:N"),
        color=alt.condition(
            self.occupation_filter,
            alt.Color("occupation:N", scale=alt.Scale(scheme='category20')),
            alt.value("lightgray")),
    ).properties(width=300, height=300, selection=self.occupation_filter)

    self.age_filter = alt.selection_multi(fields=["age"])
    self.age_chart = alt.Chart().mark_bar().encode(
        x="count()",
        y=alt.Y("age:N"),
        color=alt.condition(
            self.age_filter,
            alt.Color("age:N", scale=alt.Scale(scheme='category20')),
            alt.value("lightgray")),
    ).properties(width=500, height=500, selection=self.age_filter)
```

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
