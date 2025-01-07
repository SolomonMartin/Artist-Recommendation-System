from pathlib import Path
import scipy
import pandas as pd


def load_user_artists(user_artists_file: Path) -> scipy.sparse.csr_matrix:
    """Load the user artists and return a user-artists matirx in csr format 
        csr = compressed sparse row matrix
    """

    user_artists = pd.read_csv(user_artists_file, sep="\t")
    user_artists.set_index(["userID","artistID"], inplace=True)
    # coo = A sparse matrix in COOrdinate format.
    coo = scipy.sparse.coo_matrix(
        (
            user_artists.weight.astype(float),
            (
                user_artists.index.get_level_values(0),
                user_artists.index.get_level_values(1),
            ),
        )
    )
    # we are converting to csr as we are going to use a library that has an als using csr matrix as input
    return coo.tocsr()


class ArtistRetriever:
    # we get the artist name from the the artist id

    def __init__(self):
        self._artists_df = None

    def get_artist_name_from_id(self,artist_id: int) -> str:
        return self._artists_df.loc[artist_id,"name"]

    def load_artists(self, artists_file: Path) -> None:
        artists_df = pd.read_csv(artists_file, sep="\t")
        artists_df = artists_df.set_index("id")
        self._artists_df = artists_df


