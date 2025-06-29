"""
ui
==

Modifications from the original 'single_image_viewer.py'
--------------------------------------------------------

- Heatmap support added
    - added `heatmap()` function in global namespace
    - added `heatmap()` method in `SingleImageViewer`
    - added `VizMode.PLOT_HEAT` mode
- log axis support added
    - added `AxisScale` enum class
    - added `axis_scale_x` and `axis_scale_y` properties to `SingleImageViewer`
    - added `X` and `Y` keys to toggle axis scale in `ui()`
    - added `setup_axis_scale()` calls to `ui()`

LICENSE
-------

This file includes modified codes from:

- Erik Härkönen's 'PyViewer' library (licensed under CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/): https://github.com/harskish/pyviewer.git
"""

import ctypes
import importlib.util
import multiprocessing as mp
import random
import string
import sys
import time
import warnings
from enum import Enum
from threading import Thread

import light_process as lp
import numpy as np
import torch
from imgui_bundle import imgui, implot
# NOTE: Import GLFW from imgui_bundle.glfw_utils to avoid multile definitions
from imgui_bundle.glfw_utils import glfw  # type: ignore
from pyviewer import gl_viewer
from pyviewer.utils import (PannableArea, begin_inline, normalize_image_data,
                            reshape_grid)


def is_tensor(obj):
    return "torch" in sys.modules and torch.is_tensor(obj)


if not importlib.util.find_spec("torch"):
    def is_tensor(obj): return False


class ImgShape(ctypes.Structure):
    _fields_ = [('h', ctypes.c_uint), ('w', ctypes.c_uint), ('c', ctypes.c_uint)]


class WindowSize(ctypes.Structure):
    _fields_ = [('w', ctypes.c_uint), ('h', ctypes.c_uint)]


class VizMode(Enum):
    IMAGE = 0
    PLOT_LINE = 1
    PLOT_LINE_DOT = 2
    PLOT_DOT = 3
    PLOT_HEAT = 4


class AxisScale(Enum):
    linear = 0
    log = 1

    def get_enum(self):
        if self == AxisScale.linear:
            return implot.Scale_.linear
        elif self == AxisScale.log:
            return implot.Scale_.log10
        else:
            raise ValueError(f'Unknown AxisScale: {self}')


class SingleImageViewer:
    def __init__(self, title, key=None, hdr=False, normalize=True, vsync=True, hidden=False, pannable=True):
        self.title = title
        self.key = key or ''.join(random.choices(string.ascii_letters, k=100))
        self.ui_process = None
        self.vsync = vsync
        self.hdr = hdr
        self.normalize = normalize if not hdr else False

        # Images are copied to minimize critical section time.
        # With uint8 (~4x faster copies than float32), this is
        # faster than waiting for OpenGL upload (for some reason...)
        # TODO: HDR uses float16 framebuffer, could save 50% with halfs
        self.dtype = 'float32' if hdr else 'uint8'

        # Shared resources for inter-process communication
        # One shared 8k rgb buffer allocated (max size), subset written to
        # Size does not affect performance, only memory usage
        self.max_size_img = (2*3840, 2*2160, 3)
        ctype = ctypes.c_uint8 if self.dtype == 'uint8' else ctypes.c_float
        self.shared_buffer_img = mp.Array(ctype, np.prod(self.max_size_img).item())

        # Plotting mode: max size 1M floats per axis
        self.max_size_plot = 100_000_000
        self.shared_buffer_plot = mp.Array(ctypes.c_float, 2*self.max_size_plot)
        self.plot_marker_size = mp.Value('f', 2.0)

        # Non-scalar type: not updated in single transaction
        # Protected by shared_buffer's lock
        self.latest_shape = mp.Value(ImgShape, *(0, 0, 0), lock=False)
        self.latest_plot_len = mp.Value(ctypes.c_uint, 0, lock=False)  # single axis

        # Current window size, protected by lock
        self.curr_window_size = mp.Value(WindowSize, *(0, 0))

        # Scalar values updated atomically
        self.should_quit = mp.Value('i', 0)
        self.has_new_img = mp.Value('i', 0)

        # Image or plot
        self.viz_mode = mp.Value('i', VizMode.IMAGE.value)

        # Axis scale for plots
        self.axis_scale_x = mp.Value('i', AxisScale.linear.value)
        self.axis_scale_y = mp.Value('i', AxisScale.linear.value)

        # For hiding/showing window
        self.hidden = mp.Value(ctypes.c_bool, hidden, lock=False)

        # For enabling/disabling mouse pan and zoom
        self.pan_enabled = mp.Value(ctypes.c_bool, pannable, lock=False)

        # Pausing (via pause key on keyboard) speeds up computation
        self.paused = mp.Value(ctypes.c_bool, False, lock=False)

        # For waiting until process has started
        self.started = mp.Value(ctypes.c_bool, False, lock=False)

        self._start()

    # Called from main thread, waits until viewer is visible
    def wait_for_startup(self, timeout=10):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout and not self.started.value:
            time.sleep(1/10)
        if not self.started.value:
            print(f'SIV: wait_for_startup timed out ({timeout}s)')

    # Called from main thread, loops until viewer window is closed
    def wait_for_close(self):
        while self.ui_process.is_alive():
            time.sleep(0.2)

    @property
    def window_size(self):
        with self.curr_window_size.get_lock():
            return (self.curr_window_size.w, self.curr_window_size.h)

    def hide(self):
        self.hidden.value = True

    def show(self, sync=False):
        self.hidden.value = False
        if sync:
            self.wait_for_startup()

    def _start(self):
        self.started.value = False
        self.ui_process = lp.LightProcess(target=self.process_func)  # won't import __main__
        self.ui_process.start()

    def restart(self):
        if self.ui_process.is_alive():
            self.ui_process.join()
        self._start()

    def close(self):
        self.should_quit.value = 1
        self.ui_process.join()

    def process_func(self):
        # Avoid double cuda init issues
        # (single image viewer cannot use GPU memory anyway)
        gl_viewer.has_pycuda = False
        gl_viewer.cuda_synchronize = lambda: None

        v = gl_viewer.viewer(self.title, swap_interval=int(self.vsync), hidden=self.hidden.value)
        v._window_hidden = self.hidden.value
        v.set_interp_nearest()
        v.pan_handler = PannableArea(force_mouse_capture=True)
        compute_thread = Thread(target=self.compute, args=[v])

        def set_glfw_callbacks(window):
            self.window_size_callback(window, *glfw.get_window_size(window))  # call once to set defaults
            glfw.set_window_close_callback(window, self.window_close_callback)
            glfw.set_window_size_callback(window, self.window_size_callback)
            v.pan_handler.set_callbacks(window)
        v.start(self.ui, [compute_thread], set_glfw_callbacks)

    def window_close_callback(self, window):
        self.started.value = False

    def window_size_callback(self, window, w, h):
        with self.curr_window_size.get_lock():
            self.curr_window_size.w = w
            self.curr_window_size.h = h

    # Called from main thread
    def draw(self, img_hwc=None, img_chw=None, ignore_pause=False):
        # Paused or closed
        if (self.paused.value and not ignore_pause) or not self.ui_process.is_alive():
            return

        # Activate image mode
        self.viz_mode.value = VizMode.IMAGE.value

        assert img_hwc is not None or img_chw is not None, 'Must provide img_hwc or img_chw'
        assert img_hwc is None or img_chw is None, 'Cannot provide both img_hwc and img_chw'

        if is_tensor(img_hwc):
            img_hwc = img_hwc.detach().cpu().numpy()

        if is_tensor(img_chw):
            img_chw = img_chw.detach().cpu().numpy()

        # Convert chw to hwc, if provided
        # If grayscale: no conversion needed
        if img_chw is not None:
            img_hwc = img_chw if img_chw.ndim == 2 else np.transpose(img_chw, (1, 2, 0))
            img_chw = None

        # Convert data to valid range
        if self.normalize:
            img_hwc = normalize_image_data(img_hwc, self.dtype)
        elif not self.hdr:
            # Unnormalized LDR: must clip to avoid broken colors
            maxval = 255 if self.dtype == 'uint8' else 1.0
            img_hwc.clip(max=maxval)

        sz = np.prod(img_hwc.shape)
        assert sz <= np.prod(self.max_size_img), f'Image too large, max size {self.max_size_img}'

        # Synchronize
        with self.shared_buffer_img.get_lock():
            arr_np = np.frombuffer(self.shared_buffer_img.get_obj(), dtype=self.dtype)
            arr_np[:sz] = img_hwc.reshape(-1)
            self.latest_shape.h = img_hwc.shape[0]
            self.latest_shape.w = img_hwc.shape[1]
            self.latest_shape.c = img_hwc.shape[2]
            self.has_new_img.value = 1

    # Called from main thread
    def plot(self, y, *, x=None, xlim=None, ylim=None, ignore_pause=False):
        # Paused or closed
        if (self.paused.value and not ignore_pause) or not self.ui_process.is_alive():
            return

        # Activate plotting mode
        if self.viz_mode.value in [VizMode.IMAGE.value, VizMode.PLOT_HEAT.value]:
            self.viz_mode.value = VizMode.PLOT_LINE.value

        assert x is not None or y is not None, 'Must provide data for x or y axis'

        if is_tensor(x):
            x = x.detach().cpu().numpy()

        if is_tensor(y):
            y = y.detach().cpu().numpy()

        if xlim is None:
            xlim = (0.0, 0.0)
        if ylim is None:
            ylim = (0.0, 0.0)

        # Convert lists to np arrays
        x = np.asarray(x) if x is not None else None
        y = np.asarray(y) if y is not None else None

        # Flatten, fill missing data with linspace
        x = x.reshape(-1) if x is not None else np.linspace(0, 1, len(y.reshape(-1)))
        y = y.reshape(-1) if y is not None else np.linspace(0, 1, len(x.reshape(-1)))
        assert len(x) == len(y), 'X and Y length differs'

        sz = np.prod(x.shape)

        sz_total = sz + 4  # additional values for xlim and ylim
        assert sz_total <= self.max_size_plot, f'Too much data, max size {self.max_size_plot} per axis'

        # Synchronize
        with self.shared_buffer_plot.get_lock():
            arr_np = np.frombuffer(self.shared_buffer_plot.get_obj(), dtype='float32')
            arr_np[0:sz] = x.astype(np.float32)
            arr_np[sz:2*sz] = y.astype(np.float32)
            arr_np[2*sz:2*sz+4] = np.array([xlim[0], xlim[1], ylim[0], ylim[1]], dtype='float32')
            self.latest_plot_len.value = sz

    # Called from main thread
    def heatmap(self, x, *, h_bounds: tuple[float, float] | None = None, w_bounds: tuple[float, float] | None = None, ignore_pause=False):
        # Paused or closed
        if (self.paused.value and not ignore_pause) or not self.ui_process.is_alive():
            return

        # Activate heatmap mode
        if self.viz_mode.value != VizMode.PLOT_HEAT.value:
            self.viz_mode.value = VizMode.PLOT_HEAT.value

        assert x.ndim == 2, 'Heatmap data must be 2D array'

        if is_tensor(x):
            x = x.detach().cpu().numpy()

        h, w = x.shape

        if h_bounds is None:
            h_bounds = (0.0, float(h))
        if w_bounds is None:
            w_bounds = (0.0, float(w))

        # Convert data to float32 contiguous array
        x = np.ascontiguousarray(x, dtype='float32')
        sz = np.prod(x.shape)

        # Flatten
        x = x.reshape(-1)

        # Attach additional data for bounds
        x = np.concatenate([x, np.array((h, w, *h_bounds, *w_bounds), dtype='float32')], axis=0)
        x = np.ascontiguousarray(x, dtype='float32')
        assert len(x) == sz + 6, 'Heatmap data must have 6 additional values for bounds'
        assert sz + 6 <= self.max_size_plot, f'Too much data {sz + 6}, max size {self.max_size_plot}'

        # Synchronize
        with self.shared_buffer_plot.get_lock():
            arr_np = np.frombuffer(self.shared_buffer_plot.get_obj(), dtype='float32')
            arr_np[:sz + 6] = x.astype(np.float32)
            self.latest_plot_len.value = sz

    # Called in loop from ui thread

    def ui(self, v):
        if self.should_quit.value == 1:
            glfw.set_window_should_close(v._window, True)
            return

        # Visibility changed
        if self.hidden.value != v._window_hidden:
            v._window_hidden = self.hidden.value
            if v._window_hidden:
                glfw.hide_window(v._window)
            else:
                glfw.show_window(v._window)

        if v.keyhit(glfw.KEY_PAUSE):
            self.paused.value = not self.paused.value
        if v.keyhit(glfw.KEY_M):
            self.viz_mode.value = (self.viz_mode.value + 1) % len(VizMode)  # loop through modes
        if v.keyhit(glfw.KEY_X):
            self.axis_scale_x.value = (self.axis_scale_x.value + 1) % len(AxisScale)
        if v.keyhit(glfw.KEY_Y):
            self.axis_scale_y.value = (self.axis_scale_y.value + 1) % len(AxisScale)

        imgui.set_next_window_size(glfw.get_window_size(v._window))
        imgui.set_next_window_pos((0, 0))
        begin_inline('Output', inputs=True)

        viz_mode = VizMode(self.viz_mode.value)

        # Draw provided image
        if viz_mode == VizMode.IMAGE:
            v.pan_handler.zoom_enabled = True
            if self.pan_enabled.value:
                tex_in = v._images.get(self.key)
                if tex_in:
                    tex_H, tex_W, _ = tex_in.shape
                    cW, cH = map(int, imgui.get_content_region_avail())
                    # cW, cH = [int(r-l) for l,r in zip(
                    #    imgui.get_window_content_region_min(), imgui.get_window_content_region_max())]
                    scale = min(cW / tex_W, cH / tex_H)
                    out_res = (int(tex_W*scale), int(tex_H*scale))
                    tex = v.pan_handler.draw_to_canvas(tex_in.tex, *out_res, cW, cH)
                    imgui.image(tex, (cW, cH))
            else:
                v.draw_image(self.key, width='fit')

        # Draw heatmap
        elif viz_mode == VizMode.PLOT_HEAT:
            v.pan_handler.zoom_enabled = True
            with self.shared_buffer_plot.get_lock():
                sz = self.latest_plot_len.value
                data = np.frombuffer(self.shared_buffer_plot.get_obj(), dtype='float32', count=sz+2+4).copy()
                x = data[0:sz]
                h = int(data[sz])
                w = int(data[sz+1])
                x = x.reshape((h, w))
                h_min = float(data[sz+2])
                h_max = float(data[sz+3])
                w_min = float(data[sz+4])
                w_max = float(data[sz+5])

            W, H = glfw.get_window_size(v._window)
            style = imgui.get_style()
            avail_h = H - 2*style.window_padding.y
            avail_w = W - 2*style.window_padding.x

            psd_cmap = 'plasma'  # default colormap
            cmap = getattr(implot.Colormap_, psd_cmap, None)
            if cmap is not None:
                implot.push_colormap(cmap.value)

            if implot.begin_plot('##siv_main_heatmap', size=(avail_w, avail_h), flags=implot.Flags_.equal.value):
                scale_min = np.min(x)
                scale_max = np.max(x)

                implot.plot_heatmap(
                    '',
                    values=x,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    bounds_min=(w_min, h_min),
                    bounds_max=(w_max, h_max),
                    label_fmt=''
                )

                imgui.same_line()
                implot.colormap_scale("Amplitude", scale_min, scale_max)

                implot.end_plot()

            if cmap is not None:
                implot.pop_colormap()

        # Draw plot
        else:
            v.pan_handler.zoom_enabled = False
            x = y = None
            xlim = ylim = None

            with self.shared_buffer_plot.get_lock():
                sz = self.latest_plot_len.value
                data = np.frombuffer(self.shared_buffer_plot.get_obj(), dtype='float32', count=2*sz+4).copy()

                end_idx = 2*sz

                x = data[0:sz]
                y = data[sz:end_idx]

                xlim = (data[end_idx], data[end_idx + 1])
                ylim = (data[end_idx + 2], data[end_idx + 3])

            W, H = glfw.get_window_size(v._window)
            style = imgui.get_style()
            avail_h = H - 2*style.window_padding.y
            avail_w = W - 2*style.window_padding.x
            if implot.begin_plot('##siv_main_plot', size=(avail_w, avail_h)):
                if xlim[0] != xlim[1]:
                    implot.setup_axis_limits(implot.ImAxis_.x1, xlim[0], xlim[1], imgui.Cond_.always.value)
                if ylim[0] != ylim[1]:
                    implot.setup_axis_limits(implot.ImAxis_.y1, ylim[0], ylim[1], imgui.Cond_.always.value)

                if viz_mode in [VizMode.PLOT_LINE, VizMode.PLOT_LINE_DOT]:
                    implot.setup_axis_scale(implot.ImAxis_.x1, AxisScale(self.axis_scale_x.value).get_enum())
                    implot.setup_axis_scale(implot.ImAxis_.y1, AxisScale(self.axis_scale_y.value).get_enum())
                    implot.plot_line('', x, y)
                if viz_mode in [VizMode.PLOT_DOT, VizMode.PLOT_LINE_DOT]:
                    implot.set_next_marker_style(size=self.plot_marker_size.value)
                    implot.plot_scatter('', x, y)
                implot.end_plot()

        if self.paused.value:
            imgui.push_font(v._imgui_fonts[30])
            dl = imgui.get_window_draw_list()
            dl.add_rect_filled((5, 8), (115, 43), imgui.get_color_u32_rgba(0, 0, 0, 1))
            dl.add_text(20, 10, imgui.get_color_u32_rgba(1, 1, 1, 1), 'PAUSED')
            imgui.pop_font()
            time.sleep(1/20)  # <= real speedup of pausing comes from here

        imgui.end()

    # Called in loop from compute thread
    # Image upload loop, not needed for plotting
    def compute(self, v):
        self.started.value = True
        while not v.quit:
            if self.has_new_img.value == 1:
                with self.shared_buffer_img.get_lock():
                    shape = (self.latest_shape.h, self.latest_shape.w, self.latest_shape.c)
                    img = np.frombuffer(self.shared_buffer_img.get_obj(), dtype=self.dtype, count=np.prod(shape)).copy()
                    self.has_new_img.value = 0

                img = img.reshape(shape)
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=-1)

                v.upload_image_np(self.key, img)
            elif self.paused.value:
                time.sleep(1/10)  # paused
            else:
                time.sleep(1/80)  # idle


# Suppress warning due to LightProcess
if not sys.warnoptions:  # allow overriding with `-W` option
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy',
                            message="'pyviewer.single_image_viewer' found in sys.modules.*")

# Single global instance
# Removes need to pass variable around in code
# Just call draw() (optionally call init first)
inst: SingleImageViewer = None


def init(*args, sync=True, **kwargs):
    global inst

    if inst is None:
        inst = SingleImageViewer(*args, **kwargs)
        if sync:
            inst.wait_for_startup()  # if calling from debugger: need to give process time to start

# No-op if already open, therwise (re)start


def show_window():
    # Elif to avoid immediate restart if first init
    if inst is None:
        init('SIV')
    elif not inst.started.value:
        inst.restart()
        inst.paused.value = False
    elif inst.hidden:
        inst.show(sync=True)


def draw(*, img_hwc=None, img_chw=None, ignore_pause=False):
    init('SIV')  # no-op if init already performed
    inst.draw(img_hwc, img_chw, ignore_pause)


def grid(*, img_nchw=None, ignore_pause=False):
    init('SIV')  # no-op if init already performed
    inst.draw(img_hwc=reshape_grid(img_nchw=img_nchw), ignore_pause=ignore_pause)


def plot(y, *, x=None, xlim=None, ylim=None, ignore_pause=False):
    init('SIV')  # no-op if init already performed
    inst.plot(x=x, y=y, xlim=xlim, ylim=ylim, ignore_pause=ignore_pause)


def heatmap(x, *,  h_bounds=None, w_bounds=None, ignore_pause=False):
    init('SIV')  # no-op if init already performed
    inst.heatmap(x, h_bounds=h_bounds, w_bounds=w_bounds, ignore_pause=ignore_pause)


def set_marker_size(size):
    "Set implot marker size"
    init('SIV')
    inst.plot_marker_size.value = size
