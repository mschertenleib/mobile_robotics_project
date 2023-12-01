import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import numpy as np
import cv2


def main():
    dpg.create_context()
    dpg.configure_app(docking=True, docking_space=True, init_file='imgui.ini', auto_save_init_file=True)
    dpg.create_viewport(title='Robot interface', width=640, height=480)

    demo.show_demo()

    x = np.linspace(0, 1, 960)
    y = np.linspace(0, 1, 720)
    xv, yv = np.meshgrid(x, y)
    z = np.dstack((np.sin(xv * 10) * np.cos(yv * 14) * 0.5 + 0.5, np.sin(xv * 5) * np.cos(yv * 4) * 0.5 + 0.5,
                   np.sin(xv * 5) * np.cos(yv * 4) * 0.5 + 0.5, np.ones(xv.shape[0:2])))
    dpg.add_texture_registry(label="Texture Container", tag="__texture_container")
    dpg.add_dynamic_texture(z.shape[1], z.shape[0], z.flatten().tolist(),
                            parent="__texture_container", tag="__grid")

    with dpg.window(label="Main window", width=800, height=800, pos=(100, 100)):
        dpg.add_image("__grid")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()

    dpg.delete_item("__texture_container")
    dpg.destroy_context()


if __name__ == '__main__':
    main()
