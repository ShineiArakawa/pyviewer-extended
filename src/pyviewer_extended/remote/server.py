import base64
import contextlib
import io
import os
import signal
import threading
import uuid

import fastapi
import numpy as np

import pyviewer_extended.remote.schema as schema
import pyviewer_extended.remote.ui as siv

# ----------------------------------------------------------------------------
# Utility functions


def decode_ndarray(base64_str: str | None) -> np.ndarray | None:
    if base64_str is None:
        return None

    try:
        decoded = base64.b64decode(base64_str)

        buffer = io.BytesIO(decoded)
        array = np.load(buffer)
    except Exception as e:
        raise ValueError(f"Failed to decode ndarray: {e}")

    return array


# ----------------------------------------------------------------------------
# API setup

lock = threading.Lock()


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("Starting the server ... ")
    yield
    print("Shutting down the server ... ")

app = fastapi.FastAPI(
    title='Single Image Viewer Extended',
    lifespan=lifespan,
)


@app.post('/draw', response_model=schema.DrawResponse)
async def draw(request: schema.DrawRequest) -> schema.DrawResponse:
    img_hwc = request.img_hwc
    img_chw = request.img_chw

    if img_hwc is None and img_chw is None:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail='Either img_hwc or img_chw must be provided.'
        )

    img_chw = decode_ndarray(img_chw)
    img_hwc = decode_ndarray(img_hwc)

    with lock:
        siv.draw(
            img_chw=img_chw,
            img_hwc=img_hwc,
            ignore_pause=True
        )

    state_id = uuid.uuid4().hex

    return schema.DrawResponse(state_id=state_id)


@app.post('/grid', response_model=schema.GridResponse)
async def grid(request: schema.GridRequest) -> schema.GridResponse:
    img_nchw = request.img_nchw

    if img_nchw is None:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail='img_nchw must be provided.'
        )

    img_nchw = decode_ndarray(img_nchw)

    with lock:
        siv.grid(img_nchw=img_nchw, ignore_pause=True)

    state_id = uuid.uuid4().hex

    return schema.GridResponse(state_id=state_id)


@app.post('/plot', response_model=schema.PlotResponse)
async def plot(request: schema.PlotRequest) -> schema.PlotResponse:
    y = request.y
    x = request.x

    if y is None:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail='y must be provided.'
        )

    y = decode_ndarray(y)
    x = decode_ndarray(x)

    with lock:
        siv.plot(y=y, x=x, ignore_pause=True)

    state_id = uuid.uuid4().hex

    return schema.PlotResponse(state_id=state_id)


@app.post('/heatmap', response_model=schema.HeatmapResponse)
async def heatmap(request: schema.HeatmapRequest) -> schema.HeatmapResponse:
    x = request.x
    h_bounds = request.h_bounds
    w_bounds = request.w_bounds

    if x is None:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail='x must be provided.'
        )

    if h_bounds is not None and len(h_bounds) != 2:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail='h_bounds must be a list of two floats.'
        )

    if w_bounds is not None and len(w_bounds) != 2:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail='w_bounds must be a list of two floats.'
        )

    x = decode_ndarray(x)

    with lock:
        siv.heatmap(x=x, h_bounds=h_bounds, w_bounds=w_bounds, ignore_pause=True)

    state_id = uuid.uuid4().hex

    return schema.HeatmapResponse(state_id=state_id)


@app.get('/shutdown')
async def shutdown() -> fastapi.Response:
    """Shutdown the server."""

    os.kill(os.getpid(), signal.SIGTERM)

    return fastapi.Response(
        content='Server is shutting down.'
    )
