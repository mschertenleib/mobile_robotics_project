import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import numpy as np
import cv2


def main():
    dpg.create_context()
    dpg.configure_app(docking=True, docking_space=True, init_file='imgui.ini', auto_save_init_file=True)
    dpg.create_viewport(title='Robot interface', width=640, height=480)

    demo.show_demo()

    dpg.add_texture_registry(label="Texture Container", tag="__texture_container")
    width = 640
    height = 480
    print(width * height)
    dpg.add_dynamic_texture(width, height, np.zeros((height, width, 4), dtype=np.float32).flatten().tolist(),
                            parent="__texture_container", tag="__grid")

    with dpg.window(label="Main window", width=800, height=800, pos=(100, 100)):
        dpg.add_image("__grid")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        if cap.isOpened():
            ret, frame = cap.read()
            rgba = np.empty((frame.shape[0], frame.shape[1], 4), dtype=np.float32)
            rgba[:, :, 2::-1] = frame / 255.0
            rgba[:, :, 3] = 1.0
            if ret:
                dpg.set_value("__grid", rgba.flatten().tolist())
                cv2.imshow('main', frame)

        dpg.render_dearpygui_frame()


    cap.release()
    dpg.delete_item("__texture_container")
    dpg.destroy_context()


if __name__ == '__main__':
    main()
