import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import numpy as np


def main():
    dpg.create_context()
    dpg.configure_app(docking=True, docking_space=True, init_file='imgui.ini', auto_save_init_file=True)
    dpg.create_viewport(title='Robot interface', width=1280, height=720)

    demo.show_demo()

    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, 1000)
    xv, yv = np.meshgrid(x, y)
    z = np.dstack((np.sin(xv * 10) * np.cos(yv * 14) * 0.5 + 0.5, np.sin(xv * 5) * np.cos(yv * 4) * 0.5 + 0.5,
                   np.sin(xv * 5) * np.cos(yv * 4) * 0.5 + 0.5, np.ones(xv.shape[0:2])))
    dpg.add_texture_registry(label="Texture Container", tag="__texture_container")
    dpg.add_dynamic_texture(z.shape[0], z.shape[1], z.flatten().tolist(),
                            parent="__texture_container", tag="__grid")

    with dpg.window(label="Main window", width=800, height=800, pos=(100, 100)):
        dpg.add_image("__grid")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()

    dpg.delete_item("__texture_container")
    dpg.destroy_context()


if __name__ == '__main__':
    main()
