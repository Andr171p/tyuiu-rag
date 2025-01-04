from httpx import Response


def is_ok(response: Response) -> bool:
    return 200 <= response.status_code < 300
