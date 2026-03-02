import asyncio
import aiohttp
import os
import hashlib
from tqdm import tqdm

BASE_URL = "https://rutgers.box.com/shared/static/"

# Box_File_ID -> Expected_SHA1 for chekcsum
FILE_DATA = {
    "MIMIC": [
        (
            "7l8n3tn9tr0602tdss1x7e3uliahlibp",
            "f3adc384962136eb93d5e12c73d1e2c742387df3",
        ),
        (
            "zco48rvz5dog72970679foen6hct15c8",
            "b8db02f3e490c94e8b8b1e5c50bd127c0c340a65",
        ),
        (
            "x22qpmelx6sz3wgkm5qyc0eis429361f",
            "b32a27d7be4c8919b72a85b0bdafd716b0243f14",
        ),
        (
            "xj25sqnluiz6s4z8tzzm5phk00ohp6e8",
            "8e97e3811c42c6d5311d2a5151652be2b5b087d7",
        ),
        (
            "dxus2lsoop02chaspnwipwrf0g4wmenr",
            "947ac26f686f3068cb726b3d158fb3469c203f68",
        ),
        (
            "rts6sj441laenm2sy1qcemg7ke4om3j6",
            "8b6ab2773a3c7135d8cd71c9c33d7a480a1531e5",
        ),
        (
            "vor4hjllld7a0c3nzef8uptbb4ut3koo",
            "1648b9c50cf4b4949582955fa517c0a036982e03",
        ),
        (
            "a2qg2p4ebyrooji3z88djlokji65tlf3",
            "4c9534c904d71cefafc6e8892d0eaeb6f6990113",
        ),
        (
            "uh6kbiuqgnib5wakiv6o35gkpusyamc7",
            "b7ad32b67abebec81861253ee1e1efac66a9527f",
        ),
        (
            "h6eyhkkx48pf3ce3th1clwj43hn98j5c",
            "29fcc37cd04a9d099cb3a7566b07a3175b36e87f",
        ),
        (
            "e93dp94hxpkas45yc59n289s2wvkafgi",
            "70e585b97f2dc72130ea1c34ff0ba936a86e8e2c",
        ),
        (
            "iuvyuw7dmlxvbjvt53dj49wqn3gelqni",
            "5e4f38cc64cdb7938b144664f4863086bdf547f8",
        ),
        (
            "qxx6tjz8c3778601ib3icu6o1rranmc7",
            "6acc91136ac53e232b4c83a6b3363b3077af2407",
        ),
        (
            "ip2ninwqj8437l9fyffjprnk90ptnx9k",
            "67b20543d8240ba6db7cf696638edf21c76ab26e",
        ),
        (
            "yrtbo0lg8mjhaw624iw9bbhk1obbocwd",
            "7e3edd06e61365b481233c60890309b43d57be29",
        ),
        (
            "wmzndowgfa5xi3tvtqahxkld3ngdyjds",
            "837559cc18349e610d7950394e11be2d86d559c3",
        ),
    ],
    "Vital": [
        (
            "vtxoksmn7emeaxypb2prywgwscuefoqa",
            "3e25f5f89e77b5619f911376f714facd1d14b95e",
        ),
        (
            "euzkek7c3xoy62jisheuxqar7z5y8xig",
            "1ba93c9c4f1189f940be89db513b306ddbe93ffa",
        ),
        (
            "49lngo0benxfjw193jnqz9tctlyb3qam",
            "612ab2056288183bb2bba9724d6a66c18b15e71c",
        ),
        (
            "jf4fwgkmhry20mf5tcg9t0wxvky64um0",
            "f10ee6fe9f36b292ed5b313e6ea57357aa596c76",
        ),
        (
            "2lgxysbskfuapsaan4jypvmm8316fdkc",
            "9a7c332ae6954a0e613b38b3ee522ab109e84736",
        ),
        (
            "x27ktb4qsx43razwo4tjmxq9v1ro0x3y",
            "8d7104a83e797d5c7026b0b1492d0f91fda6b426",
        ),
        (
            "q0t36fikgf3pimhvnerwwnovfr0umtp8",
            "d60553c8cb46657f00f37c31494c5cde4b170a2b",
        ),
        (
            "ihckx2g0f981g5yz2x8v5rgwndl6yebw",
            "a64d6c051fe3f2bc0f6b438d6b9bcad7150e9c6f",
        ),
        (
            "y8j14h8tvi5b3du8nap9dnura1omfrk6",
            "d98fb31b44364875c379fa82293ca14845b9f1c2",
        ),
        (
            "fu0m9tx33jkxywq32shh0g8dg3not15u",
            "45c8a5bd810b1b5e70a63c3b9300490737fcabf4",
        ),
    ],
}

DOWNLOADS = []
for dataset, files in FILE_DATA.items():
    for i, (file_id, expected_hash) in enumerate(files, start=1):
        ext = f"{i:03d}"
        filename = f"PulseDB_{dataset}.zip.{ext}"
        url = f"{BASE_URL}{file_id}.{ext}"
        DOWNLOADS.append((filename, url, expected_hash))


def verify_checksum(filename, expected_hash):
    if not os.path.exists(filename):
        return False

    sha1 = hashlib.sha1()
    with open(filename, "rb") as f:
        while chunk := f.read(1024 * 64):  # 64KB chunks
            sha1.update(chunk)

    return sha1.hexdigest() == expected_hash


async def download_file(
    session, filename, url, expected_hash, semaphore, overall_bar, max_retries=3
):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                headers = {}
                file_mode = "wb"
                existing_size = 0

                # Check for existing file to attempt a resume
                if os.path.exists(filename):
                    existing_size = os.path.getsize(filename)
                    if existing_size > 0:
                        headers["Range"] = f"bytes={existing_size}-"
                        file_mode = "ab"

                async with session.get(url, headers=headers) as response:

                    if response.status == 416:
                        # 416 means it's likely already downloaded. We'll skip straight to hash validation.
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

                        # Stream to disk with per-file progress bar
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

                # Checksum Validation
                if verify_checksum(filename, expected_hash):
                    overall_bar.update(1)
                    return  # Success, exit retry loop
                else:
                    # If invalid, delete the corrupted file and trigger a retry
                    os.remove(filename)
                    raise ValueError("Checksum mismatch. Corrupted file deleted.")

            except (aiohttp.ClientError, OSError, ValueError) as e:
                tqdm.write(
                    f"[{filename}] Error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    overall_bar.update(1)
                    tqdm.write(
                        f"[{filename}] Failed to download securely after {max_retries} attempts."
                    )


async def main():
    semaphore = asyncio.Semaphore(5)
    connector = aiohttp.TCPConnector(limit_per_host=5)

    with tqdm(total=len(DOWNLOADS), desc="Overall", unit="file") as overall_bar:
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                download_file(
                    session, filename, url, expected_hash, semaphore, overall_bar
                )
                for filename, url, expected_hash in DOWNLOADS
            ]
            await asyncio.gather(*tasks)


if __name__ == "__main__":

    print("Starting parallel downloads with checksum verification...")
    asyncio.run(main())
    print("All tasks finished.")
