import time

import dearpygui.dearpygui as dpg
import numpy as np

MAP_HEIGHT_PX = 320
MAP_WIDTH_PX = 640


def calculate_image_plot_border_size(plot, plot_width_0, plot_height_0, axes_aspect_ratio_0):
    map_aspect_ratio = MAP_WIDTH_PX / MAP_HEIGHT_PX

    print(dpg.get_axis_limits('tag_map_axis_x'), dpg.get_axis_limits('tag_map_axis_y'))
    return

    # Get the new plot size and aspect ratio of the axes
    plot_width_1, plot_height_1 = dpg.get_item_rect_size(plot)
    axis_x_min_1, axis_x_max_1 = dpg.get_axis_limits('tag_map_axis_x')
    axis_x_range_1 = axis_x_max_1 - axis_x_min_1
    if axis_x_range_1 == 0:
        return
    axis_y_min_1, axis_y_max_1 = dpg.get_axis_limits('tag_map_axis_y')
    axis_y_range_1 = axis_y_max_1 - axis_y_min_1
    if axis_y_range_1 == 0:
        return
    axes_aspect_ratio_1 = axis_x_range_1 / axis_y_range_1

    if axes_aspect_ratio_1 == axes_aspect_ratio_0:
        return

    # Setup and solve the system of equations
    A = np.array([[1, -axes_aspect_ratio_1 * map_aspect_ratio], [1, -axes_aspect_ratio_0 * map_aspect_ratio]])
    b = np.array([plot_width_1 - axes_aspect_ratio_1 * map_aspect_ratio * plot_height_1,
                  plot_width_0 - axes_aspect_ratio_0 * map_aspect_ratio * plot_height_0])

    border_width, border_height = np.linalg.inv(A) * b
    print(border_width, border_height)


def resize_plot_to_fit_window(window, plot):
    window_width, window_height = dpg.get_item_rect_size(window)
    if window_width <= 0 or window_height <= 0:
        return

    dpg.fit_axis_data('tag_map_axis_x')
    dpg.fit_axis_data('tag_map_axis_y')

    # Use the item position within the window to deduce the window's border width and title bar height
    plot_pos_x, plot_pos_y = dpg.get_item_pos(plot)
    window_content_width = window_width - 2 * plot_pos_x
    window_content_height = window_height - plot_pos_y - plot_pos_x
    # dpg.set_item_width(plot, window_content_width)
    # dpg.set_item_height(plot, window_content_height)

    # Assume too wide:
    #
    # new_height = height + dh
    # 

    plot_width, plot_height = dpg.get_item_rect_size(plot)

    # dpg.set_axis_limits('tag_map_axis_x', 0, MAP_WIDTH_PX)
    # dpg.set_axis_limits('tag_map_axis_y', 0, MAP_HEIGHT_PX)
    print(dpg.get_axis_limits('tag_map_axis_x'), dpg.get_axis_limits('tag_map_axis_y'))


def main():
    dpg.create_context()

    dpg.configure_app(docking=True, docking_space=True, init_file='imgui.ini', auto_save_init_file=True)
    dpg.create_viewport(title='Robot interface', width=960, height=640)

    data_t = []
    data_x = []
    data_y = []

    with dpg.texture_registry():
        default_map_data = np.random.random((MAP_HEIGHT_PX, MAP_WIDTH_PX, 4)).astype(np.float32)
        default_map_data[:, :, 3] = 1.0
        dpg.add_raw_texture(width=MAP_WIDTH_PX, height=MAP_HEIGHT_PX, default_value=default_map_data,
                            format=dpg.mvFormat_Float_rgba, tag='tag_map_texture')

    with dpg.window(label='Plots', tag='tag_plot_window', no_close=True):
        dpg.add_slider_int(label='Samples to plot', tag='tag_samples_slider', default_value=300, min_value=10,
                           max_value=1200)
        dpg.add_checkbox(label='Auto-fit axes', tag='tag_checkbox_autofit', default_value=True)
        with dpg.plot(label='XY plot', tag='tag_plot_xy', width=-1, height=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label='Time [s]', tag='tag_plot_axis_x')
            with dpg.plot_axis(dpg.mvYAxis, label='Position [mm]', tag='tag_plot_axis_y'):
                dpg.add_line_series([], [], label='X', tag='tag_series_x')
                dpg.add_line_series([], [], label='Y', tag='tag_series_y')

    with dpg.window(label='Map', tag='tag_map_window', no_close=True):
        with dpg.plot(label='Map', tag='tag_map_plot', equal_aspects=True, width=-1, height=-1):
            dpg.add_plot_axis(dpg.mvXAxis, label='X [mm]', tag='tag_map_axis_x')
            with dpg.plot_axis(dpg.mvYAxis, label='Y [mm]', tag='tag_map_axis_y'):
                dpg.add_image_series(texture_tag='tag_map_texture', tag='tag_map_image_series', bounds_min=(0, 0),
                                     bounds_max=(MAP_WIDTH_PX, MAP_HEIGHT_PX))

    dpg.setup_dearpygui()
    dpg.show_viewport()

    start_time = time.time()

    while dpg.is_dearpygui_running():

        data_t.append(time.time() - start_time)
        data_x.append(np.random.random())
        data_y.append(np.random.random())

        if len(data_t) >= 2:
            num_samples_to_plot = dpg.get_value('tag_samples_slider')
            num_samples_to_plot = min(num_samples_to_plot, len(data_t))
            data_t_arr = np.array(data_t)
            data_x_arr = np.array(data_x)
            data_y_arr = np.array(data_y)

            dpg.set_value('tag_series_x', [data_t_arr.tolist(), data_x_arr.tolist()])
            dpg.set_value('tag_series_y', [data_t_arr.tolist(), data_y_arr.tolist()])

            if dpg.get_value('tag_checkbox_autofit'):
                dpg.set_axis_limits('tag_plot_axis_x', data_t_arr.item(-num_samples_to_plot), data_t_arr.item(-1))

                min_pos = min(np.min(data_x_arr[-num_samples_to_plot:]), np.min(data_y_arr[-num_samples_to_plot:]))
                max_pos = max(np.max(data_x_arr[-num_samples_to_plot:]), np.max(data_y_arr[-num_samples_to_plot:]))
                pos_range = max_pos - min_pos
                ymin = min_pos - pos_range * 0.1 if pos_range > 0 else min_pos - 1
                ymax = max_pos + pos_range * 0.1 if pos_range > 0 else max_pos + 1
                dpg.set_axis_limits('tag_plot_axis_y', ymin, ymax)
            else:
                dpg.set_axis_limits_auto('tag_plot_axis_x')
                dpg.set_axis_limits_auto('tag_plot_axis_y')

        resize_plot_to_fit_window('tag_map_window', 'tag_map_plot')
        # calculate_image_plot_border_size('tag_map_plot', plot_width_0, plot_height_0, axes_aspect_ratio_0)

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == '__main__':
    main()
