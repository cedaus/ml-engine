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
      genres = self.genre_cols
      def get_random_genre(gs):
        active = [genre for genre, g in zip(genres, gs) if g==1]
        if len(active) == 0:
          return 'Other'
        return np.random.choice(active)
      def get_all_genres(gs):
        active = [genre for genre, g in zip(genres, gs) if g==1]
        if len(active) == 0:
          return 'Other'
        return '-'.join(active)
      self.movies['genre'] = [
          get_random_genre(gs) for gs in zip(*[self.movies[genre] for genre in genres])]
      self.movies['all_genres'] = [
          get_all_genres(gs) for gs in zip(*[self.movies[genre] for genre in genres])]

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

    def split_dataframe(self, holdout_fraction=0.1):
      """
      Splits a DataFrame into training and test sets.
      Args:
        df: a dataframe.
        holdout_fraction: fraction of dataframe rows to use in the test set.
      Returns:
        train: dataframe for training
        test: dataframe for testing
      """
      self.test_df = self.df.sample(frac=holdout_fraction, replace=False)
      self.train_df = self.df[~self.df.index.isin(self.test_df.index)]

    def filtered_histogram(self, field, label, filter):
      """Creates a layered chart of histograms.
      The first layer (light gray) contains the histogram of the full data, and the
      second contains the histogram of the filtered data.
      Args:
        field: the field for which to generate the histogram.
        label: String label of the histogram.
        filter: an alt.Selection object to be used to filter the data.
      """
      base = alt.Chart().mark_bar().encode(
          x=alt.X(field, bin=alt.Bin(maxbins=10), title=label),
          y="count()",
      ).properties(
          width=300,
      )
      return alt.layer(
          base.transform_filter(filter),
          base.encode(color=alt.value('lightgray'), opacity=alt.value(.7)),
      ).resolve_scale(y='independent')


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

    def explore_movies_data(self):
        self.mark_genres()
        self.movies_ratings = self.movies.merge(
            self.ratings
            .groupby('movie_id', as_index=False)
            .agg({'rating': ['count', 'mean']})
            .flatten_cols(),
            on='movie_id')

        (self.movies_ratings[['title', 'rating count', 'rating mean']]
         .sort_values('rating count', ascending=False)
         .head(10))

        (self.movies_ratings[['title', 'rating count', 'rating mean']]
         .mask('rating count', lambda x: x > 20)
         .sort_values('rating mean', ascending=False)
         .head(10))

        self.genre_filter = alt.selection_multi(fields=['genre'])
        self.genre_chart = alt.Chart().mark_bar().encode(
            x="count()",
            y=alt.Y('genre'),
            color=alt.condition(
                self.genre_filter,
                alt.Color("genre:N"),
                alt.value('lightgray'))
        ).properties(height=300, selection=self.genre_filter)
