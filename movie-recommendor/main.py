from __future__ import print_function

import numpy as np
import pandas as pd
import collections
from mpl_toolkits.mplot3d import Axes3D
from IPython import display
from matplotlib import pyplot as plt
import sklearn
import sklearn.manifold
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# Add some convenience functions to Pandas DataFrame.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format
def mask(df, key, function):
  """Returns a filtered dataframe, by applying function to key"""
  return df[function(df[key])]

def flatten_cols(df):
  df.columns = [' '.join(col).strip() for col in df.columns.values]
  return df

pd.DataFrame.mask = mask
pd.DataFrame.flatten_cols = flatten_cols

# Install Altair and activate its colab renderer.
print("Installing Altair...")
!pip install git+git://github.com/altair-viz/altair.git
import altair as alt
alt.data_transformers.enable('default', max_rows=None)
alt.renderers.enable('colab')
print("Done installing Altair.")

# Install spreadsheets and import authentication module.
USER_RATINGS = False
!pip install --upgrade -q gspread
from google.colab import auth
import gspread
from oauth2client.client import GoogleCredentials

# 1. Explore Dataset and get data frame
# 2. Split the data frame into train and test df
# 3.


"""
Loading data to get dataframes
"""
DLE = DataLoaderExplorer();
df = DLE.load_movielens_dataset();
# Split the Data Frame in Train and Test data frame (90% and 10% of actual data frame)
DLE.split_dataframe()
print(M.train_df)
print(M.test_df)

"""
Exploring user data
"""
DLE.explore_user_data();
print(DLE.users)
print(DLE.users_ratings)
DLE.users.describe(include=[np.object])

# Create a chart for occupation and age
alt.hconcat(
    DLE.occupation_chart,
    DLE.age_chart,
    data=DLE.users_ratings
)
# Create a chart for the count, and one for the mean.
alt.hconcat(
    DLE.filtered_histogram('rating count', '# ratings / user', DLE.occupation_filter),
    DLE.filtered_histogram('rating mean', 'mean user rating', DLE.occupation_filter),
    DLE.occupation_chart,
    data=DLE.users_ratings
)

"""
Exploring movies data
"""
DLE.explore_movies_data()
print(DLE.movies)
print(DLE.movies_ratings)
# Display the number of ratings and average rating per movie.
alt.hconcat(
    DLE.filtered_histogram('rating count', '# ratings / movie', DLE.genre_filter),
    DLE.filtered_histogram('rating mean', 'mean movie rating', DLE.genre_filter),
    DLE.genre_chart,
    data=DLE.movies_ratings
)

"""
"""
SP = SimilarityPrediction(measure='DOT', users=DLE.users, movies=DLE.movies, ratings=DLE.ratings)
EV = EmbeddingVizualizer(movies=DLE.movies, movies_ratings=DLE.movies_ratings)

"""
Setingup Matrix Factorization
Find train and test sparse matrix
"""
MF = MatrixFactorization(all_df=DLE.df, train_df=DLE.train_df, test_df=DLE.test_df, users_count= DLE.users.shape[0], movies_count=DLE.movies.shape[0]);
MF.find_sparse_matrices()

"""
Building collaborative filtering model
"""
def build_model(mf, embedding_dim=3, init_stddev=1.):

  # Initializing Users and Movies random embeddings
  random_embeddings = mf.initializing_random_embeddings(
      sparse_matrix = mf.sparse_matrices['train'],
      embedding_dim = embedding_dim,
      init_stddev = init_stddev
  )

  # Calculating Train and Test Loss
  mse_errors = mf.find_sparse_mean_square_errors(embeddings=random_embeddings)

  losses = {
    'train_error_observed': mse_errors['train'],
    'test_error_observed': mse_errors['test'],
  }

  total_loss = mse_errors['train']

  # Building Model
  model = CFModel(
      embedding_vars=random_embeddings,
      loss=total_loss,
      metrics = [losses]
  )

  return model

"""
# Training Model
"""
model = build_model(mf=MF)
model.train(num_iterations=1500, learning_rate=10.)

"""
Similarity Prediction
"""
SP.user_recommendations(embeddings=model.embeddings, user_id=555, k=10)
SP.movie_neighbors(embeddings=model.embeddings, title_substring='Maya', k=10)

"""
# Training Model with hypertuned parameters
"""
model_lowinit = build_model(mf=MF, embedding_dim=30, init_stddev=0.05)
model_lowinit.train(num_iterations=1000, learning_rate=10.)

"""
Similarity Prediction for Model with hypertuned parameters
"""
SP.user_recommendations(embeddings=model_lowinit.embeddings, user_id=555, k=10)
SP.movie_neighbors(embeddings=model_lowinit.embeddings, title_substring='Maya', k=10)

"""
Building regularized model
"""
def build_regularized_model(mf, embedding_dim=3, init_stddev=1., regularization_coeff=.1, gravity_coeff=1.):
    # Initializing Users and Movies random embeddings
    random_embeddings = mf.initializing_random_embeddings(
        sparse_matrix = mf.sparse_matrices['train'],
        embedding_dim = embedding_dim,
        init_stddev = init_stddev
    )

    U = random_embeddings['users']
    V = random_embeddings['movies']

    # Calculating Train and Test Loss
    mse_errors = mf.find_sparse_mean_square_errors(embeddings=random_embeddings)

    # Calculating regularization_loss
    regularization_loss = regularization_coeff * (
      tf.reduce_sum(U*U)/U.shape[0].value + tf.reduce_sum(V*V)/V.shape[0].value)

    # Calculating gravity_loss
    gravity_loss = gravity_coeff * 1. / (U.shape[0].value*V.shape[0].value) * tf.reduce_sum(
      tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))

    #
    total_loss = mse_errors['train'] + regularization_loss + gravity_loss

    losses = {
      'train_error_observed': mse_errors['train'],
      'test_error_observed': mse_errors['test'],
    }
    loss_components = {
      'observed_loss': mse_errors['train'],
      'regularization_loss': regularization_loss,
      'gravity_loss': gravity_loss,
    }

    #
    model = CFModel(
    embedding_vars=random_embeddings,
    loss=total_loss,
    metrics=[losses, loss_components]
    )
    return model

"""
Training Regularized Model
"""
reg_model = build_regularized_model(
    mf=MF,
    regularization_coeff=0.1,
    gravity_coeff=1.0,
    embedding_dim=35,
    init_stddev=.05
)
reg_model.train(num_iterations=2000, learning_rate=20.)

"""
"""
EV.movie_embedding_norm([model, model_lowinit])


# Visualize the regularized model embeddings
EV.tsne_movie_embeddings(embeddings=reg_model.embeddings)
