def get_barbell_curl():
    # Define the angle ranges for wrist-elbow-vertical alignment
    _ANGLE_WRIST_ELBOW_VERT = {
        'PASS': (10, 50),
        'TRANS': (75, 109),
        'NORMAL': (110, 145)
    }

    # Configure the settings for the barbell curl exercise
    settings = {
        'REF_ANGLE': _ANGLE_WRIST_ELBOW_VERT,
        'HIP_THRESH': 10,
        'SHOULDER_THRESH': 25,
        # 'ELBOW_THRESH': 145,  # Uncomment if needed
        'OFFSET_THRESH': 80.0,
        'INACTIVE_THRESH': 10.0,
        'CNT_FRAME_THRESH': 50,
        'FEEDBACK_ID_MAP': {
            
            0: {
                'msg': 'KEEP YOUR BACK STRAIGHT',
                'pos': (30, 170),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)
            },
            1: {
                'msg': 'AVOID EXCESSIVE SWING',
                'pos': (30, 125),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)
            }
        }
    }

    return settings


def get_bent_over_dumbbell_row():
    # Define the angle ranges for elbow-hip-shoulder alignment
    _ANGLE_ELBOW_HIP_SHLDR = {
        'PASS': (0, 9),
        'TRANS': (10, 29),
        'NORMAL': (30, 55)
    }

    # Configure the settings for the bent-over dumbbell row exercise
    settings = {
        'REF_ANGLE': _ANGLE_ELBOW_HIP_SHLDR,
        'HIP_THRESH': 40,
        'ANKLE_THRESH': 45,
        'SHLDR_THRESH': 145,
        'OFFSET_THRESH': 55.0,
        'INACTIVE_THRESH': 10.0,
        'CNT_FRAME_THRESH': 50,
        'FEEDBACK_ID_MAP': {
            0: {
                'msg': 'LOWER YOUR TORSO',
                'pos': (30, 80),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)
            },
            1: {
                'msg': 'AVOID KNEES OVER TOES',
                'pos': (30, 170),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)
            },
            2: {
                'msg': 'KEEP YOUR BACK STRAIGHT',
                'pos': (30, 200),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)
            }
        }
    }

    return settings


def get_squat_with_weights():
    # Define the angle ranges for hip-knee-vertical alignment
    _ANGLE_HIP_KNEE_VERT = {
        'PASS': (80, 95),
        'TRANS': (35, 65),
        'NORMAL': (0, 32)
    }

    # Configure the settings for the weighted squat exercise
    settings = {
        'REF_ANGLE': _ANGLE_HIP_KNEE_VERT,
        'KNEE_THRESH': [70, 80, 90],
        'HIP_THRESH': [10, 50],
        'ANKLE_THRESH': 45,
        'OFFSET_THRESH': 55.0,
        'INACTIVE_THRESH': 10.0,
        'CNT_FRAME_THRESH': 50,
        'FEEDBACK_ID_MAP': {
            0: {
                'msg': 'RISE UP',
                'pos': (30, 80),
                'text_color': (0, 0, 0),
                'text_color_bg': (255, 255, 0)
            },
            1: {
                'msg': 'AVOID KNEES OVER TOES',
                'pos': (30, 170),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)
            },
            2: {
                'msg': 'SQUAT IS TOO DEEP',
                'pos': (30, 125),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)
            }
        }
    }

    return settings
