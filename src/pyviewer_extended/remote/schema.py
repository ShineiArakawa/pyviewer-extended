import pydantic


class DrawRequest(pydantic.BaseModel):
    img_hwc: str | None = None
    img_chw: str | None = None


class DrawResponse(pydantic.BaseModel):
    state_id: str


class GridRequest(pydantic.BaseModel):
    img_nchw: str


class GridResponse(pydantic.BaseModel):
    state_id: str


class PlotRequest(pydantic.BaseModel):
    y: str
    x: str | None = None
    xlim: list[float] | None = None
    ylim: list[float] | None = None


class HeatmapRequest(pydantic.BaseModel):
    x: str
    h_bounds: list[float] | None = None
    w_bounds: list[float] | None = None


class HeatmapResponse(pydantic.BaseModel):
    state_id: str


class PlotResponse(pydantic.BaseModel):
    state_id: str
