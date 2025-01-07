from contextlib import (
    asynccontextmanager,
    AbstractAsyncContextManager
)

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AbstractAsyncContextManager[None]:
    yield
