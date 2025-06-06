"""This is a demo script for 'single_image_viewer' extended with remote capabilities.

Make sure to run the viewer server beforehand using the command:
```bash
python -m pyviewer_extended.remote --host localhost --port 13110
```

Then, you can run this script to interact with the server.
"""

import numpy as np

import pyviewer_extended.remote.siv as siv

# Initialize the client to connect to the remote server
# By default, it connects to 'localhost' on port 13110. Therefore, if your server listens on that port, you do not need to call this function.
siv.init_client(address='localhost', port=13110)

# # ----------------------------------------------------------------------------------------------------
# # Draw a single image

# img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

# siv.draw(img_hwc=img)

# # ----------------------------------------------------------------------------------------------------
# # Next, draw multiple images in a single window

# if input('Next, draw a grid? (y/n): ').strip().lower() == 'y':
#     # Create a grid of images

#     dtype = np.float32

#     x = np.linspace(0.0, 1.0, 5)
#     y = np.linspace(0.0, 1.0, 5)

#     tl_color = np.array([1.0, 0.0, 0.0], dtype=dtype).reshape((3, 1, 1))  # Red
#     tr_color = np.array([0.0, 1.0, 0.0], dtype=dtype).reshape((3, 1, 1))  # Green
#     br_color = np.array([0.0, 0.0, 1.0], dtype=dtype).reshape((3, 1, 1))  # Blue
#     bl_color = np.array([1.0, 1.0, 0.0], dtype=dtype).reshape((3, 1, 1))  # Yellow

#     img = np.ones((3, 256, 256), dtype=dtype)
#     grid = np.zeros((len(y) * len(x), *img.shape), dtype=dtype)

#     for i_x in range(len(x)):
#         for i_y in range(len(y)):
#             u = x[i_x]
#             v = y[i_y]

#             tl_weight = (1.0 - u) * (1.0 - v)
#             tr_weight = u * (1.0 - v)
#             br_weight = u * v
#             bl_weight = (1.0 - u) * v

#             grid[i_y * len(x) + i_x] = (
#                 tl_weight * tl_color +
#                 tr_weight * tr_color +
#                 br_weight * br_color +
#                 bl_weight * bl_color
#             ) * img

#     siv.grid(img_nchw=grid)


# # ----------------------------------------------------------------------------------------------------
# # Next, plot a graph

# if input('Next, plot a graph? (y/n): ').strip().lower() == 'y':
#     # Create a simple line plot

#     x = np.linspace(0, 1, 10_000)
#     y = np.sin(2 * np.pi * x)

#     siv.plot(y=y, x=x)

# print('Done.')

# # ----------------------------------------------------------------------------------------------------
# # Next, draw a power spectrum density

# if input('Next, draw a power spectrum density? (y/n): ').strip().lower() == 'y':
#     # Create a power spectrum density plot

#     t = np.linspace(0.0, 1.0, 64, dtype=np.float32)
#     x = np.sin(2 * np.pi * 4 * t)
#     img = np.tile(x[np.newaxis, :], (64, 1))

#     siv.psd(img)

# ----------------------------------------------------------------------------------------------------
# Finally, plot a radial profile

# if input('Next, plot a radially averaged power spectrum density? (y/n): ').strip().lower() == 'y':
# Create a radial profile plot

t = np.linspace(0.0, 1.0, 64, dtype=np.float32)
x = np.sin(2 * np.pi * 4 * t)
img = np.tile(x[np.newaxis, :], (64, 1))
img = img[np.newaxis, :, :]  # Add channel dimension
img = np.concatenate((img, img), axis=0)  # Duplicate to make it 2D
print(img.shape)  # Should be (2, 64, 64)
siv.radial_psd(img)
