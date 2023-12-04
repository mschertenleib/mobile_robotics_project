is_doing_local_nav = 0


@onevent
def requestspeed(left_speed, right_speed):
    global is_doing_local_nav, motor_left_target, motor_right_target

    if is_doing_local_nav == 0:
        motor_left_target = left_speed
        motor_right_target = right_speed
