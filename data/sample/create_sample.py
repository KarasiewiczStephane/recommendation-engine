"""Script to create a sample subset of MovieLens 100K for CI testing.

Generates synthetic data files in the same format as the real dataset
to enable fast testing without network access.
"""

import random
from pathlib import Path


def create_sample_data(output_dir: str = "data/sample/ml-100k") -> None:
    """Generate sample MovieLens data files for testing.

    Args:
        output_dir: Directory to write the sample files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    # Generate sample ratings (u.data format: user_id\titem_id\trating\ttimestamp)
    ratings_lines = []
    for user_id in range(1, 21):
        n_ratings = random.randint(5, 15)
        items = random.sample(range(1, 51), n_ratings)
        for item_id in items:
            rating = random.randint(1, 5)
            timestamp = 880000000 + random.randint(0, 1000000)
            ratings_lines.append(f"{user_id}\t{item_id}\t{rating}\t{timestamp}")

    (out / "u.data").write_text("\n".join(ratings_lines) + "\n")

    # Generate sample movies (u.item format)
    genres = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    item_lines = []
    for item_id in range(1, 51):
        title = f"Movie {item_id} (1990)"
        release_date = "01-Jan-1990"
        video_release = ""
        imdb_url = f"http://example.com/{item_id}"
        genre_flags = [str(random.randint(0, 1)) for _ in genres]
        # Ensure at least one genre
        if all(g == "0" for g in genre_flags):
            genre_flags[random.randint(0, len(genres) - 1)] = "1"
        fields = [
            str(item_id),
            title,
            release_date,
            video_release,
            imdb_url,
        ] + genre_flags
        item_lines.append("|".join(fields))

    (out / "u.item").write_text("\n".join(item_lines) + "\n")

    print(
        f"Created sample data with {len(ratings_lines)} ratings across 20 users and 50 items"
    )


if __name__ == "__main__":
    create_sample_data()
