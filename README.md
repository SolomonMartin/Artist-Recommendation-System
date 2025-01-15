# Artist Recommender System

This project implements a music recommendation system. The system utilizes collaborative filtering to recommend artists to users based on their listening history. 

## Features
- **Collaborative Filtering**: Recommends artists using implicit feedback and matrix factorization.
- **Customizable**: Supports any model compatible with the `implicit` library.
- **Efficient Retrieval**: Retrieves artist names from IDs for human-readable recommendations.

## Project Structure
```
.
├── main.py         # Main script to run the recommender system
├── data.py         # Data loading and artist retrieval utilities
├── Data/           # Directory for input data files
│   ├── user_artists.dat  # User-artist interactions
│   └── artists.dat       # Artist metadata
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Files

- `user_artists.dat`: Tab-separated file containing user-artist interaction data.
- `artists.dat`: Tab-separated file containing artist metadata.

### Example Formats
#### user_artists.dat
```
userID	artistID	weight
1	52	13883
1	53	11690
```

#### artists.dat
```
id	name
52	Coldplay
53	The Beatles
```

## Usage

1. Ensure the data files (`user_artists.dat` and `artists.dat`) are located in the `Data/` directory.

2. Run the main script:
   ```bash
   python main.py
   ```

3. Example output:
   ```
   Artist Name 1: 0.95
   Artist Name 2: 0.88
   ```

## Code Overview

### `main.py`
- **`ImplicitRecommender`**: Main class for training the model and generating recommendations.
- **`recommend`**: Method to retrieve top-N artist recommendations for a user.

### `data.py`
- **`load_user_artists`**: Loads user-artist interaction data into a sparse matrix.
- **`ArtistRetriever`**: Maps artist IDs to artist names using metadata.

## Dependencies
- `implicit`
- `pandas`
- `scipy`

Install these via `pip install -r requirements.txt`.

## Future Enhancements
- Add support for additional recommendation models.
- Implement a web-based interface for easier interaction.
- Enhance error handling for missing or invalid data.

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, feel free to reach out or submit a pull request!
