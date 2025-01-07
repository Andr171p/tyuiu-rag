import aiofiles
from pathlib import Path


def load_txt(file_path: Path | str) -> str:
    with open(
        file=file_path,
        mode="r",
        encoding="utf-8"
    ) as file:
        return file.read()


async def load_txt_async(file_path: Path | str) -> str:
    async with aiofiles.open(
        file=file_path,
        mode="r",
        encoding="utf-8"
    ) as file:
        return await file.read()
