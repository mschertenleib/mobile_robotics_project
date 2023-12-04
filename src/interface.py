import cv2
import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import numpy as np


def main():
    dpg.create_context()
    dpg.configure_app(docking=True, docking_space=True, init_file='imgui.ini', auto_save_init_file=True)
    dpg.create_viewport(title='Robot interface', width=640, height=480)

    demo.show_demo()

    frame_width = 960
    frame_height = 720
    rgba = np.empty((frame_height, frame_width, 4), dtype=np.float32)

    with dpg.texture_registry():
        dpg.add_raw_texture(width=frame_width, height=frame_height, default_value=rgba, format=dpg.mvFormat_Float_rgba,
                            tag="frame")

    with dpg.window(label="Main window", width=800, height=800, pos=(100, 100)):
        dpg.add_image("frame")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rgba[:, :, 2::-1] = frame / 255.0
                rgba[:, :, 3] = 1.0
                dpg.set_value("frame", rgba.flatten())

        dpg.render_dearpygui_frame()

    cap.release()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
