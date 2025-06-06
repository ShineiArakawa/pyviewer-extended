# isort: skip_file
# autopep8: off
from __future__ import annotations

import base64
import io
import urllib.parse as urlparse

import torch

import httpx
import numpy as np
import pyviewer.single_image_viewer as siv

import pyviewer_extended.remote.schema as schema
# autopep8: on

# ----------------------------------------------------------------------------
# Global variables

_url: str = "http://localhost:13110"

_timeout: float = 10.0  # seconds


# ----------------------------------------------------------------------------
# Utility functions


def encode_ndarray(array: np.ndarray | None) -> str | None:
    if array is None:
        return None

    try:
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode ndarray: {e}")

    return encoded

# ----------------------------------------------------------------------------
# Client functions


def init_client(address: str = "localhost", port: int = 13110, timeout: float = 10.0) -> None:
    """Initialize the client to connect to the remote server.
    Call this function only when your server listens on a different address or port than the default "localhost:13110".

    Parameters
    ----------
    address : str, optional
        Address of the Single Image Viewer server, defaults to "localhost"
    port : int, optional
        Port number of the Single Image Viewer server, defaults to 13110
    """

    global _url
    _url = f"http://{address}:{port}"

    global _timeout
    _timeout = timeout


def draw(
    *,
    \
    img_hwc: np.ndarray | torch.Tensor | None = None,
    img_chw: np.ndarray | torch.Tensor | None = None
) -> str | None:
    """Draw an image using the remote Single Image Viewer server.
    This is equivalent to calling `siv.draw(...)` but sends the image to a remote server instead of drawing it locally.

    Parameters
    ----------
    img_hwc : np.ndarray | torch.Tensor | None, optional
        Image in HWC format (height, width, channels), defaults to None
    img_chw : np.ndarray | torch.Tensor | None, optional
        Image in CHW format (channels, height, width), defaults to None

    Returns
    -------
    str | None
        The state ID of the drawn image, or None if the request failed.
    """

    if siv.is_tensor(img_hwc):
        img_hwc = img_hwc.detach().cpu().numpy()
    if siv.is_tensor(img_chw):
        img_chw = img_chw.detach().cpu().numpy()

    # Encode as base64 strings
    img_hwc = encode_ndarray(img_hwc)
    img_chw = encode_ndarray(img_chw)

    # Create the request body
    body = schema.DrawRequest(
        img_hwc=img_hwc,
        img_chw=img_chw
    ).model_dump()

    # Post the request
    response = httpx.post(
        urlparse.urljoin(_url, 'draw'),
        json=body,
        timeout=_timeout
    )

    # Check for errors
    if response.status_code != 200:
        print(f'[Error] Failed to upload image - {response.status_code}: {response.text}')
        return None

    # Parse the response
    response_data = schema.DrawResponse.model_validate(response.json())

    return response_data.state_id


def grid(
    img_nchw: np.ndarray | torch.Tensor,
) -> str | None:
    """Draw a grid of images using the remote Single Image Viewer server.
    This is equivalent to calling `siv.grid(...)` but sends the images to a remote server instead of drawing them locally.

    Parameters
    ----------
    img_nchw : np.ndarray | torch.Tensor
        Grid of images in NCHW format (batch, channels, height, width)

    Returns
    -------
    str | None
        The state ID of the drawn grid, or None if the request failed.
    """

    assert img_nchw is not None, "'img_nchw' must be provided"

    if siv.is_tensor(img_nchw):
        img_nchw = img_nchw.detach().cpu().numpy()

    # Encode as base64 string
    img_nchw = encode_ndarray(img_nchw)

    # Create the request body
    body = schema.GridRequest(
        img_nchw=img_nchw
    ).model_dump()

    # Post the request
    response = httpx.post(
        urlparse.urljoin(_url, 'grid'),
        json=body,
        timeout=_timeout
    )

    # Check for errors
    if response.status_code != 200:
        print(f'[Error] Failed to upload grid - {response.status_code}: {response.text}')
        return None

    # Parse the response
    response_data = schema.GridResponse.model_validate(response.json())

    return response_data.state_id


def plot(
    y: np.ndarray | torch.Tensor,
    *,
    x: np.ndarray | torch.Tensor | None = None
) -> str | None:
    """Plot a line graph using the remote Single Image Viewer server.
    This is equivalent to calling `siv.plot(...)` but sends the data to a remote server instead of plotting it locally.

    Parameters
    ----------
    y : np.ndarray | torch.Tensor
        Y-axis data
    x : np.ndarray | torch.Tensor | None, optional
        X-axis data, defaults to None

    Returns
    -------
    str | None
        The state ID of the plotted graph, or None if the request failed.
    """

    if siv.is_tensor(y):
        y = y.detach().cpu().numpy()
    if x is not None and siv.is_tensor(x):
        x = x.detach().cpu().numpy()

    # Encode as base64 strings
    y = encode_ndarray(y)
    x = encode_ndarray(x)

    # Create the request body
    body = schema.PlotRequest(
        y=y,
        x=x
    ).model_dump()

    # Post the request
    response = httpx.post(
        urlparse.urljoin(_url, 'plot'),
        json=body,
        timeout=_timeout
    )

    # Check for errors
    if response.status_code != 200:
        print(f'[Error] Failed to plot - {response.status_code}: {response.text}')
        return None

    # Parse the response
    response_data = schema.PlotResponse.model_validate(response.json())

    return response_data.state_id


def heatmap(
    x: np.ndarray | torch.Tensor,
    *,
    h_bounds: tuple[float, float] | None = None,
    w_bounds: tuple[float, float] | None = None,
) -> str | None:
    """
    """

    assert x is not None, "'x' must be provided"
    assert x.ndim == 2, "'x' must be a 2D array (height, width)"

    if isinstance(x, np.ndarray):
        x = x.astype(np.float32)
    elif siv.is_tensor(x):
        x = x.to(dtype='float32')

    # Encode as base64 string
    x = encode_ndarray(x)

    # Create the request body
    body = schema.HeatmapRequest(
        x=x,
        h_bounds=h_bounds,
        w_bounds=w_bounds
    ).model_dump()

    # Post the request
    response = httpx.post(
        urlparse.urljoin(_url, 'heatmap'),
        json=body,
        timeout=_timeout
    )

    # Check for errors
    if response.status_code != 200:
        print(f'[Error] Failed to draw heatmap - {response.status_code}: {response.text}')
        return None

    # Parse the response
    response_data = schema.HeatmapResponse.model_validate(response.json())

    return response_data.state_id


def psd(
    img_hw: np.ndarray | torch.Tensor,
    *,
    kaiser_beta: float = 8.0,
    padding_factor: int = 4,
) -> str | None:
    """Draw a Power Spectral Density (PSD) plot using the remote Single Image Viewer server.

    This is equivalent to calling `siv.psd(...)` but sends the image to a remote server instead of drawing it locally.

    Parameters
    ----------
    img_hw : np.ndarray | torch.Tensor
        Image in HW format (height, width)
    kaiser_beta : float, optional
        Beta parameter for the Kaiser window, defaults to 8.0
    padding_factor : int, optional
        Padding factor for the PSD computation, defaults to 4
    cmap : str, optional
        Colormap to use for the PSD visualization, defaults to 'viridis'

    Returns
    -------
    str | None
        The state ID of the drawn PSD plot, or None if the request failed.
    """

    assert img_hw is not None, "'img_hw' must be provided"
    assert img_hw.ndim == 2, "'img_hw' must be a 2D array (height, width)"

    if isinstance(img_hw, np.ndarray):
        img_hw = img_hw.astype(np.float64)
    elif siv.is_tensor(img_hw):
        img_hw = img_hw.to(dtype='float64')

    import radpsd
    psd = radpsd.compute_psd(img_hw, is_db_scale=True, beta=kaiser_beta, padding_factor=padding_factor)

    if siv.is_tensor(img_hw):
        psd = psd.detach().cpu().numpy()

    psd = np.ascontiguousarray(psd, dtype=np.float32)

    h, w = img_hw.shape
    half_h = h // 2
    half_w = w // 2

    return heatmap(psd, h_bounds=(-half_h, half_h), w_bounds=(-half_w, half_w))
