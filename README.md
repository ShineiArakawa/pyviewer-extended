# PyViewer-extended

On the shoulders of giants, this package adds some additional features to the original PyViewer package.

## 1. Install

```bash
pip install git@github.com:ShineiArakawa/pyviewer-extended.git
```

```bash
uv add git+https://github.com/ShineiArakawa/pyviewer-extended.git
```

## 2. Usage

### 2.1 Multi Texture Windows

See the [example script](/examples/demo_docking_viewer_3panels.py), which creates a GUI with three isolated docking windows, each with its own texture.

### 2.2 Remote Single Image Viewer

See the [example script](/examples/demo_remote_siv.py), which creates a GUI with a server that can be accessed remotely to view a single image.

## License

This library is licensed under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

It is derived from [PyViewer](https://github.com/harskish/pyviewer.git) by Erik Härkönen, used in accordance with the ShareAlike terms.

See also [LICENSE](/LICENSE) for more details.
