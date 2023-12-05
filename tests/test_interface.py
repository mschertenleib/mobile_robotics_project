import time

import dearpygui.dearpygui as dpg
import numpy as np


def main():
    dpg.create_context()

    dpg.configure_app(docking=True, docking_space=True, init_file='imgui.ini', auto_save_init_file=True)
    dpg.create_viewport(title='Robot interface', width=1280, height=720)

    data_t = []
    data_x = []
    data_y = []

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

    dpg.setup_dearpygui()
    dpg.show_viewport()

    start_time = time.time()

    while dpg.is_dearpygui_running():
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

        data_t.append(time.time() - start_time)
        data_x.append(np.random.random())
        data_y.append(np.random.random())

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == '__main__':
    main()
