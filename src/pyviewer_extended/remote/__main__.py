
import argparse
import threading

import pyviewer.single_image_viewer as siv
import uvicorn

# ----------------------------------------------------------------------------
# Command-line interface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Single Image Viewer Client',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host address of the Single Image Viewer server',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=13110,
        help='Port number of the Single Image Viewer server',
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # ----------------------------------------------------------------------------
    # Launch the server in a separate thread
    def launch_uvicorn():
        """Launch the Uvicorn server in a separate thread."""
        uvicorn.run(
            'pyviewer_extended.remote.server:app',
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    server_thread = threading.Thread(target=launch_uvicorn, daemon=True)
    server_thread.start()

    # ----------------------------------------------------------------------------
    # Set the window visible

    siv.show_window()

    while True:
        if siv.inst is not None and not siv.inst.ui_process.is_alive():
            # This is not a graceful shutdown, but rather a forced exit.
            server_thread.join(timeout=0.1)
            break
