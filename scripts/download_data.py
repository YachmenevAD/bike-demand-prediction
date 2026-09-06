from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/275/"
    "bike+sharing+dataset.zip"
)
OUTPUT_PATH = Path("data/raw/hour.csv")
ARCHIVE_FILE_NAME = "hour.csv"


def main() -> None:
    if OUTPUT_PATH.exists():
        print(f"Data already exists: {OUTPUT_PATH}")
        return

    print(f"Downloading data from: {DATA_URL}")

    request = Request(
        DATA_URL,
        headers={"User-Agent": "bike-demand-prediction/1.0"},
    )

    with urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        csv_bytes = archive.read(ARCHIVE_FILE_NAME)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(csv_bytes)

    print(f"Data saved to: {OUTPUT_PATH}")
    print(f"Downloaded size: {len(csv_bytes):,} bytes")


if __name__ == "__main__":
    main()