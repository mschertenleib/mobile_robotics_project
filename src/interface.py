from typing import *

import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

from parameters import *
from threaded_capture import *

Tag = Union[int, str]


def resize_image_to_fit_window(window: Tag, image: Tag, image_aspect_ratio: float):
    window_width, window_height = dpg.get_item_rect_size(window)
    if window_width <= 0 or window_height <= 0:
        return

    # Use the image position within the window to deduce the window's border width and title bar height
    image_pos_x, image_pos_y = dpg.get_item_pos(image)
    window_content_width = window_width - 2 * image_pos_x
    window_content_height = window_height - image_pos_y - image_pos_x
    if window_content_width <= 0 or window_content_height <= 0:
        return

    # Make the image as big as possible while keep its aspect ratio
    image_width, image_height = window_content_width, window_content_height
    if window_content_width / window_content_height >= image_aspect_ratio:
        image_width = int(image_height * image_aspect_ratio)
    else:
        image_height = int(image_width / image_aspect_ratio)

    dpg.set_item_width(image, image_width)
    dpg.set_item_height(image, image_height)


def main():
    dpg.create_context()
    dpg.configure_app(docking=True, docking_space=True, init_file='imgui.ini', auto_save_init_file=True)

    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    img_frame_rgba_f32 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.float32)
    img_frame_rgba_f32[:, :, 3] = 1.0
    img_map_rgba_f32 = np.zeros((MAP_HEIGHT_PX, MAP_WIDTH_PX, 4), dtype=np.float32)
    img_map_rgba_f32[:, :, 3] = 1.0

    FRAME_ASPECT_RATIO = FRAME_WIDTH / FRAME_HEIGHT
    MAP_ASPECT_RATIO = MAP_WIDTH_PX / MAP_HEIGHT_PX

    with dpg.texture_registry():
        dpg.add_raw_texture(width=FRAME_WIDTH, height=FRAME_HEIGHT, default_value=img_frame_rgba_f32,
                            format=dpg.mvFormat_Float_rgba,
                            tag='tag_frame_texture')
        dpg.add_raw_texture(width=MAP_WIDTH_PX, height=MAP_HEIGHT_PX, default_value=img_map_rgba_f32,
                            format=dpg.mvFormat_Float_rgba,
                            tag='tag_map_texture')

    with dpg.window(label='Camera', tag='tag_camera_window', no_close=True):
        dpg.add_image(texture_tag='tag_frame_texture', tag='tag_frame_image')

    with dpg.window(label='Map', tag='tag_map_window', no_close=True):
        dpg.add_image(texture_tag='tag_map_texture', tag='tag_map_image')

    video_thread = VideoThread(FRAME_WIDTH, FRAME_HEIGHT)

    dpg.create_viewport(title='Robot interface', width=640, height=480)
    dpg.setup_dearpygui()
    demo.show_demo()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        is_frame_new = video_thread.get_frame(frame)
        if is_frame_new:
            img_frame_rgba_f32[:, :, 2::-1] = frame / 255.0
            dpg.set_value('tag_frame_texture', img_frame_rgba_f32.flatten())

        resize_image_to_fit_window('tag_camera_window', 'tag_frame_image', FRAME_ASPECT_RATIO)
        resize_image_to_fit_window('tag_map_window', 'tag_map_image', MAP_ASPECT_RATIO)

        dpg.render_dearpygui_frame()

    video_thread.stop()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
