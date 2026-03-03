import argparse
import asyncio
import os
import aiohttp
from tqdm import tqdm
from google.oauth2 import service_account
import google.auth.transport.requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CREDENTIALS_FILE = os.path.join(
    SCRIPT_DIR, "blood-pressure-prediction-ml-28274a212118.json"
)
FOLDER_ID = "1QZ-Z-C9MoLk1S9MovF3Tt6NjAEGbSJI3"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def list_drive_files(num_subjects):
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds)

    files = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{FOLDER_ID}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, size)",
                pageToken=page_token,
                pageSize=100,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    if num_subjects is not None:
        files = files[:num_subjects]

    return files


def get_access_token():
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


async def download_file(
    session, file_id, filename, remote_size, token, semaphore, overall_bar, max_retries=3
):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

    async with semaphore:
        # Skip if already fully downloaded
        if remote_size is not None and os.path.exists(filename):
            if os.path.getsize(filename) == remote_size:
                overall_bar.update(1)
                return

        for attempt in range(max_retries):
            try:
                headers = {"Authorization": f"Bearer {token}"}
                file_mode = "wb"
                existing_size = 0

                if os.path.exists(filename):
                    existing_size = os.path.getsize(filename)
                    if existing_size > 0:
                        headers["Range"] = f"bytes={existing_size}-"
                        file_mode = "ab"

                async with session.get(url, headers=headers) as response:
                    if response.status == 416:
                        # Already fully downloaded
                        pass
                    else:
                        response.raise_for_status()

                        # If the server ignores the Range header, overwrite from the start
                        if response.status == 200 and existing_size > 0:
                            file_mode = "wb"
                            existing_size = 0

                        content_length = response.headers.get("Content-Length")
                        total = (
                            int(content_length) + existing_size
                            if content_length
                            else None
                        )

                        with tqdm(
                            total=total,
                            initial=existing_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=filename,
                            leave=False,
                        ) as pbar:
                            with open(filename, file_mode) as f:
                                async for chunk in response.content.iter_chunked(
                                    1024 * 64
                                ):
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                overall_bar.update(1)
                return

            except (aiohttp.ClientError, OSError) as e:
                tqdm.write(
                    f"[{filename}] Error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    overall_bar.update(1)
                    tqdm.write(
                        f"[{filename}] Failed to download after {max_retries} attempts."
                    )


async def main(num_subjects):
    print("Listing files from Google Drive...")
    files = list_drive_files(num_subjects)
    print(f"Downloading {len(files)} subject files.")

    token = get_access_token()

    semaphore = asyncio.Semaphore(5)
    connector = aiohttp.TCPConnector(limit_per_host=5)

    with tqdm(total=len(files), desc="Overall", unit="file") as overall_bar:
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                download_file(
                    session,
                    f["id"],
                    os.path.join(DATA_DIR, f["name"]),
                    int(f["size"]) if "size" in f else None,
                    token,
                    semaphore,
                    overall_bar,
                )
                for f in files
            ]
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download VitalDB subject files from Google Drive."
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=None,
        help="Number of subjects to download (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Starting parallel downloads...")
    asyncio.run(main(args.num_subjects))
    print("All tasks finished.")
