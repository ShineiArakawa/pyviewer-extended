import numpy as np
from imgui_bundle import imgui

import pyviewer_extended

if __name__ == '__main__':
    class ExampleViewer(pyviewer_extended.MultiTexturesDockingViewer):
        def setup_state(self):
            self.state.seed_0 = 0
            self.state.seed_1 = 0
            self.state.seed_2 = 0

        def compute(self):
            rand_0 = np.random.RandomState(seed=self.state.seed_0)
            img_0 = rand_0.randn(256, 256, 3).astype(np.float32)
            img_0 = np.clip(img_0, 0, 1)

            rand_1 = np.random.RandomState(seed=self.state.seed_1)
            img_1 = rand_1.randn(256, 256, 3).astype(np.float32)
            img_1 = np.clip(img_1, 0, 1)

            rand_2 = np.random.RandomState(seed=self.state.seed_2)
            img_2 = rand_2.randn(256, 256, 3).astype(np.float32)
            img_2 = np.clip(img_2, 0, 1)

            return {
                # NOTE: These keys must match the panel names passed when instantiating the viewer
                'Noise 0': img_0,  # HWC
                'Noise 1': img_1,  # HWC
                'Noise 2': img_2,  # HWC
            }

        @pyviewer_extended.dockable
        def toolbar(self):
            imgui.text(f'Dynamic font size: {self.fonts[0].font_size:.1f}')
            self.state.seed_0 = imgui.slider_int('Seed 0', self.state.seed_0, 0, 1000)[1]
            self.state.seed_1 = imgui.slider_int('Seed 1', self.state.seed_1, 0, 1000)[1]
            self.state.seed_2 = imgui.slider_int('Seed 2', self.state.seed_2, 0, 1000)[1]
            imgui.get_io().font_global_scale = imgui.slider_float('Font global scale', imgui.get_io().font_global_scale, 0.1, 5)[1]
            self.ui_scale = imgui.slider_float('UI scale', self.ui_scale, 0.1, 5.0)[1]

        def drag_and_drop_callback(self, paths):
            print(paths)
            return True

    _ = ExampleViewer(f'Demo Docking Viewer (3 panels)', ['Noise 0', 'Noise 1', 'Noise 2'], enable_vsync=True)
