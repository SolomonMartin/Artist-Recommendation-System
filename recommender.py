from pathlib import Path
from typing import Tuple, List

import implicit
import scipy

from data import load_user_artists, ArtistRetriever



class ImplicitRecommender:

    """
    computes recommendations for a given user using the implicit library.

    attributes:
     - artist_retirever: an ArtistRetriever instance
     - implicit_model: an implicit model

    """

    def __init__(
        self,
        artist_retirever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model


    def fit(self, user_artist_matrix: scipy.sparse.csr_matrix) -> None:
        # Fit the model to the user artists matrix
        # the implicit model gets trained here ( the whole of matrix factorization and alternating squares takes place)
        self.implicit_model.fit(user_artist_matrix)

    
    def recommend(
        self,
        user_id:int,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int, # recommending the top n artists
    ) -> Tuple[List[str], List[float]]:
        # return the top n recommendations for the given user

        artist_ids, scores = self.implicit_model.recommend( user_id, user_artists_matrix[n])

        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]

        return artists, scores


if __name__ == "__main__":

    user_artists = load_user_artists(Path("Data/user_artists.dat"))

    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("Data/artists.dat"))

    # instantiate ALS using implicit
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    # instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)

    artists, scores = recommender.recommend(5, user_artists, 5)

    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")