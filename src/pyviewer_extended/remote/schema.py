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


class PlotResponse(pydantic.BaseModel):
    state_id: str
