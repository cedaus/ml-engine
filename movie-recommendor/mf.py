from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class MatrixFactorization():
    """
    df:
    """
    def __init__(self, all_df, train_df, test_df, users_count, movies_count):
        self.df = all_df
        self.train_df = train_df
        self.test_df = test_df
        self.users_count = users_count
        self.movies_count = movies_count
        self.sparse_matrices = None

    def build_sparse_matrix(self, dataframe):
      """
      Args:
        dataframe: a pd.DataFrame with `user_id`, `movie_id` and `rating` columns.
      Returns:
        a tf.SparseTensor representing the sparse matrix b/w users and movies and its value as rating
      """
      indices = dataframe[['user_id', 'movie_id']].values
      print('Generating indices for user_id and movie_id\n', indices)
      values = dataframe['rating'].values
      print('Generating values for above indices\n', values)
      sparse_matrix = tf.SparseTensor(
          indices=indices,
          values=values,
          dense_shape=[self.users_count, self.movies_count]
          )
      print(sparse_matrix)
      return sparse_matrix

    def find_sparse_matrices(self):
      # SparseTensor representation of the train and test datasets.
      # Here A is just a representation for sparse matrix
      print('Working to get sparse matrix for train data')
      A_train = self.build_sparse_matrix(self.train_df)
      print('\nWorking to get sparse matrix for test data')
      A_test = self.build_sparse_matrix(self.test_df)

      self.sparse_matrices = {'train': A_train, 'test': A_test}
      return self.sparse_matrices

    def initializing_random_embeddings(self, sparse_matrix, embedding_dim=3, init_stddev=1):
        user_embeddings = tf.Variable(tf.random_normal(
        [sparse_matrix.dense_shape[0], embedding_dim], stddev=init_stddev))

        print(user_embeddings)

        movie_embeddings = tf.Variable(tf.random_normal(
        [sparse_matrix.dense_shape[1], embedding_dim], stddev=init_stddev))

        print(movie_embeddings)

        embeddings = {'users': user_embeddings, 'movies': movie_embeddings}
        return embeddings

    def calculate_sparse_mean_square_error(self, sparse_matrix, embeddings, method=1):
        """
          Args:
            sparse_ratings: A SparseTensor matrix, of dense_shape [users_count, movies_count]
            user_embeddings: A dense Tensor U of shape [users_count, k] where k is the embedding
              dimension, such that U_i is the embedding of user i.
            movie_embeddings: A dense Tensor V of shape [movies_count, k] where k is the embedding
              dimension, such that V_j is the embedding of movie j.
          Returns:
            A scalar Tensor representing the MSE between the true ratings and the
              model's predictions.
          """
        if method == 1:
            # Predictions = U and V Transpose
            predictions = tf.gather_nd(
            tf.matmul(embeddings['users'], embeddings['movies'], transpose_b=True), sparse_matrix.indices)
            loss = tf.losses.mean_squared_error(sparse_matrix.values, predictions)
        else:
            # Predictions = U and V Dot Product
            U = tf.gather(embeddings['users'], sparse_matrix.indices[:, 0])
            V = tf.gather(embeddings['movies'], sparse_matrix.indices[:, 1])
            predictions = tf.reduce_sum(U * V, axis=1)
            loss = tf.losses.mean_squared_error(sparse_matrix.values, predictions)
        return loss

    def find_sparse_mean_square_errors(self, embeddings):
        train_loss = self.calculate_sparse_mean_square_error(sparse_matrix=self.sparse_matrices['train'], embeddings=embeddings)
        print('Train Loss is:', train_loss)
        test_loss = self.calculate_sparse_mean_square_error(sparse_matrix=self.sparse_matrices['test'], embeddings=embeddings)
        print('Test Loss is:', test_loss)

        errors = {'train': train_loss, 'test': test_loss}
        return errors

# Collaborative Filtering Model helper class
class CFModel(object):
  """Simple class that represents a collaborative filtering model"""
  def __init__(self, embedding_vars, loss, metrics=None):
    """Initializes a CFModel.
    Args:
      embedding_vars: A dictionary of tf.Variables.
      loss: A float Tensor. The loss to optimize.
      metrics: optional list of dictionaries of Tensors. The metrics in each
        dictionary will be plotted in a separate figure during training.
    """
    self._embedding_vars = embedding_vars
    self._loss = loss
    self._metrics = metrics
    self._embeddings = {k: None for k in embedding_vars}
    self._session = None

  def plot_results(self, iterations, metrics_vals):
    num_subplots = len(self._metrics)+1
    fig = plt.figure()
    fig.set_size_inches(num_subplots*10, 8)

    for i, metric_vals in enumerate(metrics_vals):
        ax = fig.add_subplot(1, num_subplots, i+1)
        for k, v in metric_vals.items():
          ax.plot(iterations, v, label=k)
        ax.set_xlim([1, len(iterations)])
        ax.legend()

  def train(self, num_iterations=100, learning_rate=1.0, plot_results=True,
            optimizer=tf.train.GradientDescentOptimizer):
    """Trains the model.
    Args:
      iterations: number of iterations to run.
      learning_rate: optimizer learning rate.
      plot_results: whether to plot the results at the end of training.
      optimizer: the optimizer to use. Default to GradientDescentOptimizer.
    Returns:
      The metrics dictionary evaluated at the last iteration.
    """
    with self._loss.graph.as_default():
      opt = optimizer(learning_rate)
      train_op = opt.minimize(self._loss)
      local_init_op = tf.group(
          tf.variables_initializer(opt.variables()),
          tf.local_variables_initializer())
      if self._session is None:
        self._session = tf.Session()
        with self._session.as_default():
          self._session.run(tf.global_variables_initializer())
          self._session.run(tf.tables_initializer())
          tf.train.start_queue_runners()

    with self._session.as_default():
      local_init_op.run()
      iterations = []
      metrics = self._metrics or ({},)
      metrics_vals = [collections.defaultdict(list) for _ in self._metrics]

      # Train and append results.
      for i in range(num_iterations + 1):
        _, results = self._session.run((train_op, metrics))
        if (i % 10 == 0) or i == num_iterations:
          print("\r Iteration %d: " % i + ", ".join(



                ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
                end='')
          iterations.append(i)
          for metric_val, result in zip(metrics_vals, results):
            for k, v in result.items():
              metric_val[k].append(v)

      for k, v in self._embedding_vars.items():
        self._embeddings[k] = v.eval()

      print('\n Final users and movies embeddings are:\n', self._embeddings)

      if plot_results:
        self.plot_results(iterations, metrics_vals)

      print('\n Final train and test error results:\n', results)

#
class SimilarityPrediction:
    def __init__(self, measure, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def compute_scores(self, query_embedding, item_embeddings, measure='DOT'):
        u = query_embedding
        V = item_embeddings

        if measure == 'COSINE':
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            u = u / np.linalg.norm(u)

        scores = u.dot(V.T)
        return scores

    def user_recommendations(self, embeddings, user_id, exclude_rated=False, k=6, measure='DOT'):
      scores = self.compute_scores(
          query_embedding=embeddings["users"][user_id],
          item_embeddings=embeddings["movies"],
          measure=measure
      )
      score_key = measure + ' score'
      df = pd.DataFrame({
          score_key: list(scores),
          'movie_id': self.movies['movie_id'],
          'titles': self.movies['title'],
          'genres': self.movies['all_genres'],
      })

      if exclude_rated:
        # remove movies that are already rated
        rated_movies = self.ratings[self.ratings.user_id == str(user_id)]["movie_id"].values
        df = df[df.movie_id.apply(lambda movie_id: movie_id not in rated_movies)]

      display.display(df.sort_values([score_key], ascending=False).head(k))


    def movie_neighbors(self, embeddings, title_substring, k=6, measure='DOT'):
      # Search for movie ids that match the given substring.
      ids =  self.movies[self.movies['title'].str.contains(title_substring)].index.values
      titles = self.movies.iloc[ids]['title'].values
      if len(titles) == 0:
        raise ValueError("Found no movies with title %s" % title_substring)
      print("Nearest neighbors of : %s." % titles[0])
      if len(titles) > 1:
        print("[Found more than one matching movie. Other candidates: {}]".format(
            ", ".join(titles[1:])))
      movie_id = ids[0]
      scores = self.compute_scores(
          query_embedding=embeddings["movies"][movie_id],
          item_embeddings=embeddings["movies"],
          measure=measure
      )
      score_key = measure + ' score'
      df = pd.DataFrame({
          score_key: list(scores),
          'titles': self.movies['title'],
          'genres': self.movies['all_genres']
      })
      display.display(df.sort_values([score_key], ascending=False).head(k))

#
class EmbeddingVizualizer():
    def __init__(self, movies, movies_ratings):
        self.movies = movies
        self.movies_ratings = movies_ratings

    def movie_embedding_norm(self, mods):
      """Visualizes the norm and number of ratings of the movie embeddings.
      Args:
        mod: A MFModel object.
      """
      if not isinstance(mods, list):
        models = [mods]
      else:
        models = mods

      df = pd.DataFrame({
          'title': self.movies['title'],
          'genre': self.movies['genre'],
          'num_ratings': self.movies_ratings['rating count'],
      })
      charts = []
      brush = alt.selection_interval()
      for i, model in enumerate(models):
        norm_key = 'norm'+str(i)
        df[norm_key] = np.linalg.norm(model.embeddings["movies"], axis=1)
        nearest = alt.selection(
            type='single', encodings=['x', 'y'], on='mouseover', nearest=True,
            empty='none')
        base = alt.Chart().mark_circle().encode(
            x='num_ratings',
            y=norm_key,
            color=alt.condition(brush, alt.value('#4c78a8'), alt.value('lightgray'))
        ).properties(
            selection=nearest).add_selection(brush)
        text = alt.Chart().mark_text(align='center', dx=5, dy=-5).encode(
            x='num_ratings', y=norm_key,
            text=alt.condition(nearest, 'title', alt.value('')))
        charts.append(alt.layer(base, text))
      return alt.hconcat(*charts, data=df)

    def visualize_movie_embeddings(self, data, x, y):
      genre_filter = alt.selection_multi(fields=['genre'])
      genre_chart = alt.Chart().mark_bar().encode(
            x="count()",
            y=alt.Y('genre'),
            color=alt.condition(
                genre_filter,
                alt.Color("genre:N"),
                alt.value('lightgray'))
        ).properties(height=300, selection=genre_filter)

      nearest = alt.selection(
          type='single', encodings=['x', 'y'], on='mouseover', nearest=True,
          empty='none')
      base = alt.Chart().mark_circle().encode(
          x=x,
          y=y,
          color=alt.condition(genre_filter, "genre", alt.value("whitesmoke")),
      ).properties(
          width=600,
          height=600,
          selection=nearest)
      text = alt.Chart().mark_text(align='left', dx=5, dy=-5).encode(
          x=x,
          y=y,
          text=alt.condition(nearest, 'title', alt.value('')))
      return alt.hconcat(alt.layer(base, text), genre_chart, data=data)

    def tsne_movie_embeddings(self, embeddings):
      """Visualizes the movie embeddings, projected using t-SNE with Cosine measure.
      Args:
        model: A MFModel object.
      """
      tsne = sklearn.manifold.TSNE(
          n_components=2, perplexity=40, metric='cosine', early_exaggeration=10.0,
          init='pca', verbose=True, n_iter=400)

      print('Running t-SNE...')
      V_proj = tsne.fit_transform(embeddings["movies"])
      self.movies.loc[:,'x'] = V_proj[:, 0]
      self.movies.loc[:,'y'] = V_proj[:, 1]
      return self.visualize_movie_embeddings(self.movies, 'x', 'y')
