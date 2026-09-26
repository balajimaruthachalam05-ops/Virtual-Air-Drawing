import cv2
import mediapipe as mp
import time
import math
import numpy as np


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# ============================================================
# NORMAL WINDOW
# ============================================================

window_name = "Virtual Air Drawing"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    1280,
    720
)


# ============================================================
# CANVAS
# ============================================================

canvas = None

previous_x = None
previous_y = None


# ============================================================
# UNDO / REDO
# ============================================================

undo_stack = []
redo_stack = []

MAX_HISTORY = 30


def save_undo_state():

    global canvas

    if canvas is not None:

        undo_stack.append(canvas.copy())

        if len(undo_stack) > MAX_HISTORY:
            undo_stack.pop(0)

        redo_stack.clear()


def undo():

    global canvas
    global previous_x
    global previous_y
    global shape_start_x
    global shape_start_y
    global shape_preview

    if len(undo_stack) > 0:

        redo_stack.append(canvas.copy())

        canvas = undo_stack.pop()

        previous_x = None
        previous_y = None

        shape_start_x = None
        shape_start_y = None
        shape_preview = None

        print("UNDO")

    else:

        print("Nothing to undo")


def redo():

    global canvas
    global previous_x
    global previous_y
    global shape_start_x
    global shape_start_y
    global shape_preview

    if len(redo_stack) > 0:

        undo_stack.append(canvas.copy())

        canvas = redo_stack.pop()

        previous_x = None
        previous_y = None

        shape_start_x = None
        shape_start_y = None
        shape_preview = None

        print("REDO")

    else:

        print("Nothing to redo")


def clear_canvas():

    global canvas
    global previous_x
    global previous_y
    global shape_start_x
    global shape_start_y
    global shape_preview

    if canvas is not None:

        undo_stack.append(canvas.copy())

        if len(undo_stack) > MAX_HISTORY:
            undo_stack.pop(0)

        redo_stack.clear()

        canvas[:] = 0

    previous_x = None
    previous_y = None

    shape_start_x = None
    shape_start_y = None
    shape_preview = None

    print("CANVAS CLEARED")


# ============================================================
# DRAWING SETTINGS
# ============================================================

current_color = (0, 255, 255)

brush_size = 5

eraser_size = 40


# ============================================================
# DRAWING MODE
# ============================================================

current_mode = "FREEHAND"

shape_start_x = None
shape_start_y = None

shape_preview = None


# ============================================================
# TOOLBAR
# ============================================================

toolbar_height = 80


# ============================================================
# COLOR BUTTONS
# ============================================================

buttons = [

    (10, 90, (0, 255, 255), "YELLOW"),

    (100, 180, (255, 0, 0), "BLUE"),

    (190, 270, (0, 255, 0), "GREEN"),

    (280, 360, (0, 0, 255), "RED"),

    (370, 450, (255, 255, 255), "WHITE")
]


# ============================================================
# GENERAL BUTTONS
# ============================================================

eraser_button = (460, 550)

clear_button = (560, 650)

undo_button = (660, 745)

redo_button = (755, 840)

save_button = (850, 935)


# ============================================================
# SHAPE BUTTONS
# ============================================================

freehand_button = (945, 1040)

circle_button = (1050, 1145)

rectangle_button = (1155, 1250)

triangle_button = (1260, 1355)

cube_button = (1365, 1460)


# ============================================================
# BUTTON DELAY
# ============================================================

last_action_time = 0

action_delay = 0.7


# ============================================================
# INDEX FINGER DETECTION
# ============================================================

def is_index_finger_up(hand_landmarks):

    index_tip = hand_landmarks.landmark[8]

    index_pip = hand_landmarks.landmark[6]

    return index_tip.y < index_pip.y


# ============================================================
# DRAW CIRCLE
# ============================================================

def draw_circle(
    target,
    start_x,
    start_y,
    end_x,
    end_y,
    color
):

    radius = int(
        math.sqrt(
            (end_x - start_x) ** 2
            +
            (end_y - start_y) ** 2
        )
    )

    if radius > 5:

        cv2.circle(
            target,
            (start_x, start_y),
            radius,
            color,
            brush_size,
            cv2.LINE_AA
        )


# ============================================================
# DRAW RECTANGLE
# ============================================================

def draw_rectangle(
    target,
    start_x,
    start_y,
    end_x,
    end_y,
    color
):

    cv2.rectangle(
        target,
        (start_x, start_y),
        (end_x, end_y),
        color,
        brush_size,
        cv2.LINE_AA
    )


# ============================================================
# DRAW TRIANGLE
# ============================================================

def draw_triangle(
    target,
    start_x,
    start_y,
    end_x,
    end_y,
    color
):

    width = end_x - start_x
    height = end_y - start_y

    top_x = start_x + width // 2
    top_y = start_y

    left_x = start_x
    left_y = end_y

    right_x = end_x
    right_y = end_y

    points = np.array([
        [top_x, top_y],
        [left_x, left_y],
        [right_x, right_y]
    ])

    cv2.polylines(
        target,
        [points],
        True,
        color,
        brush_size,
        cv2.LINE_AA
    )


# ============================================================
# DRAW 3D CUBE
# ============================================================

def draw_cube(
    target,
    start_x,
    start_y,
    end_x,
    end_y,
    color
):

    width = end_x - start_x
    height = end_y - start_y

    offset = int(
        min(
            abs(width),
            abs(height)
        ) * 0.25
    )

    if offset < 10:
        offset = 10


    # --------------------------------------------------------
    # FRONT FACE
    # --------------------------------------------------------

    cv2.rectangle(
        target,
        (start_x, start_y),
        (end_x, end_y),
        color,
        brush_size,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # BACK FACE
    # --------------------------------------------------------

    back_x1 = start_x + offset
    back_y1 = start_y - offset

    back_x2 = end_x + offset
    back_y2 = end_y - offset

    cv2.rectangle(
        target,
        (back_x1, back_y1),
        (back_x2, back_y2),
        color,
        brush_size,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # CONNECTING LINES
    # --------------------------------------------------------

    cv2.line(
        target,
        (start_x, start_y),
        (back_x1, back_y1),
        color,
        brush_size,
        cv2.LINE_AA
    )

    cv2.line(
        target,
        (end_x, start_y),
        (back_x2, back_y1),
        color,
        brush_size,
        cv2.LINE_AA
    )

    cv2.line(
        target,
        (start_x, end_y),
        (back_x1, back_y2),
        color,
        brush_size,
        cv2.LINE_AA
    )

    cv2.line(
        target,
        (end_x, end_y),
        (back_x2, back_y2),
        color,
        brush_size,
        cv2.LINE_AA
    )


# ============================================================
# DRAW SELECTED SHAPE
# ============================================================

def draw_shape(
    target,
    start_x,
    start_y,
    end_x,
    end_y
):

    if current_mode == "CIRCLE":

        draw_circle(
            target,
            start_x,
            start_y,
            end_x,
            end_y,
            current_color
        )


    elif current_mode == "RECTANGLE":

        draw_rectangle(
            target,
            start_x,
            start_y,
            end_x,
            end_y,
            current_color
        )


    elif current_mode == "TRIANGLE":

        draw_triangle(
            target,
            start_x,
            start_y,
            end_x,
            end_y,
            current_color
        )


    elif current_mode == "CUBE":

        draw_cube(
            target,
            start_x,
            start_y,
            end_x,
            end_y,
            current_color
        )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read camera frame.")

        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # ========================================================
    # CREATE CANVAS
    # ========================================================

    if canvas is None:

        canvas = np.zeros_like(frame)


    # ========================================================
    # TOOLBAR BACKGROUND
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], toolbar_height),
        (245, 245, 245),
        -1
    )


    # ========================================================
    # COLOR BUTTONS
    # ========================================================

    for x1, x2, color, name in buttons:

        cv2.rectangle(
            frame,
            (x1, 10),
            (x2, 70),
            color,
            -1
        )

        cv2.putText(
            frame,
            name,
            (x1 + 5, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1
        )


    # ========================================================
    # GENERAL BUTTONS
    # ========================================================

    general_buttons = [

        (eraser_button, "ERASER"),

        (clear_button, "CLEAR"),

        (undo_button, "UNDO"),

        (redo_button, "REDO"),

        (save_button, "SAVE")
    ]


    for button, name in general_buttons:

        cv2.rectangle(
            frame,
            (button[0], 10),
            (button[1], 70),
            (220, 220, 220),
            -1
        )

        cv2.putText(
            frame,
            name,
            (button[0] + 7, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            2
        )


    # ========================================================
    # SHAPE BUTTONS
    # ========================================================

    shape_buttons = [

        (freehand_button, "FREE"),

        (circle_button, "CIRCLE"),

        (rectangle_button, "RECT"),

        (triangle_button, "TRI"),

        (cube_button, "CUBE")
    ]


    for button, name in shape_buttons:

        selected = False


        if name == "FREE" and current_mode == "FREEHAND":
            selected = True

        elif name == "CIRCLE" and current_mode == "CIRCLE":
            selected = True

        elif name == "RECT" and current_mode == "RECTANGLE":
            selected = True

        elif name == "TRI" and current_mode == "TRIANGLE":
            selected = True

        elif name == "CUBE" and current_mode == "CUBE":
            selected = True


        if selected:

            button_color = (160, 210, 160)

        else:

            button_color = (210, 210, 210)


        cv2.rectangle(
            frame,
            (button[0], 10),
            (button[1], 70),
            button_color,
            -1
        )

        cv2.putText(
            frame,
            name,
            (button[0] + 5, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            2
        )


    # ========================================================
    # HAND TRACKING
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)


    # ========================================================
    # HAND FOUND
    # ========================================================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape


        # ----------------------------------------------------
        # INDEX FINGERTIP
        # ----------------------------------------------------

        index_tip = hand_landmarks.landmark[8]

        x = int(index_tip.x * w)

        y = int(index_tip.y * h)


        # Keep coordinates inside frame
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))


        # ----------------------------------------------------
        # FINGER STATE
        # ----------------------------------------------------

        finger_up = is_index_finger_up(
            hand_landmarks
        )


        # ====================================================
        # TOOLBAR
        # ====================================================

        if y < toolbar_height:

            previous_x = None
            previous_y = None

            shape_start_x = None
            shape_start_y = None

            shape_preview = None

            current_time = time.time()


            if current_time - last_action_time > action_delay:

                button_used = False


                # =================================================
                # COLORS
                # =================================================

                for x1, x2, color, name in buttons:

                    if x1 <= x <= x2:

                        current_color = color

                        brush_size = 5

                        print(
                            "Selected color:",
                            name
                        )

                        last_action_time = current_time

                        button_used = True

                        break


                # =================================================
                # ERASER
                # =================================================

                if not button_used:

                    if eraser_button[0] <= x <= eraser_button[1]:

                        current_color = (0, 0, 0)

                        brush_size = eraser_size

                        current_mode = "FREEHAND"

                        print("Eraser selected")

                        last_action_time = current_time

                        button_used = True


                # =================================================
                # CLEAR
                # =================================================

                if not button_used:

                    if clear_button[0] <= x <= clear_button[1]:

                        clear_canvas()

                        last_action_time = current_time

                        button_used = True


                # =================================================
                # UNDO
                # =================================================

                if not button_used:

                    if undo_button[0] <= x <= undo_button[1]:

                        undo()

                        last_action_time = current_time

                        button_used = True


                # =================================================
                # REDO
                # =================================================

                if not button_used:

                    if redo_button[0] <= x <= redo_button[1]:

                        redo()

                        last_action_time = current_time

                        button_used = True


                # =================================================
                # SAVE
                # =================================================

                if not button_used:

                    if save_button[0] <= x <= save_button[1]:

                        cv2.imwrite(
                            "air_drawing.png",
                            canvas
                        )

                        print(
                            "Drawing saved as air_drawing.png"
                        )

                        last_action_time = current_time

                        button_used = True


                # =================================================
                # FREEHAND
                # =================================================

                if not button_used:

                    if freehand_button[0] <= x <= freehand_button[1]:

                        current_mode = "FREEHAND"

                        print("Mode: FREEHAND")

                        last_action_time = current_time

                        button_used = True


                # =================================================
                # CIRCLE
                # =================================================

                if not button_used:

                    if circle_button[0] <= x <= circle_button[1]:

                        current_mode = "CIRCLE"

                        print("Mode: CIRCLE")

                        last_action_time = current_time

                        button_used = True


                # =================================================
                # RECTANGLE
                # =================================================

                if not button_used:

                    if rectangle_button[0] <= x <= rectangle_button[1]:

                        current_mode = "RECTANGLE"

                        print("Mode: RECTANGLE")

                        last_action_time = current_time

                        button_used = True


                # =================================================
                # TRIANGLE
                # =================================================

                if not button_used:

                    if triangle_button[0] <= x <= triangle_button[1]:

                        current_mode = "TRIANGLE"

                        print("Mode: TRIANGLE")

                        last_action_time = current_time

                        button_used = True


                # =================================================
                # CUBE
                # =================================================

                if not button_used:

                    if cube_button[0] <= x <= cube_button[1]:

                        current_mode = "CUBE"

                        print("Mode: CUBE")

                        last_action_time = current_time

                        button_used = True


        # ====================================================
        # DRAWING AREA
        # ====================================================

        else:


            # =================================================
            # FREEHAND MODE
            # =================================================

            if current_mode == "FREEHAND":

                if finger_up:

                    if (
                        previous_x is None
                        or previous_y is None
                    ):

                        save_undo_state()


                    if (
                        previous_x is not None
                        and previous_y is not None
                    ):

                        cv2.line(
                            canvas,
                            (previous_x, previous_y),
                            (x, y),
                            current_color,
                            brush_size,
                            cv2.LINE_AA
                        )


                    previous_x = x
                    previous_y = y


                else:

                    previous_x = None
                    previous_y = None


            # =================================================
            # SHAPE MODE
            # =================================================

            else:

                # ---------------------------------------------
                # INDEX UP = START / PREVIEW
                # ---------------------------------------------

                if finger_up:

                    if (
                        shape_start_x is None
                        or shape_start_y is None
                    ):

                        shape_start_x = x
                        shape_start_y = y

                        shape_preview = canvas.copy()

                    else:

                        shape_preview = canvas.copy()

                        draw_shape(
                            shape_preview,
                            shape_start_x,
                            shape_start_y,
                            x,
                            y
                        )


                # ---------------------------------------------
                # INDEX DOWN = FINISH SHAPE
                # ---------------------------------------------

                else:

                    if (
                        shape_start_x is not None
                        and shape_start_y is not None
                    ):

                        save_undo_state()

                        draw_shape(
                            canvas,
                            shape_start_x,
                            shape_start_y,
                            x,
                            y
                        )

                        print(
                            "Created:",
                            current_mode
                        )

                        shape_start_x = None
                        shape_start_y = None
                        shape_preview = None


        # ====================================================
        # CURSOR
        # ====================================================

        if finger_up:

            cv2.circle(
                frame,
                (x, y),
                10,
                (0, 255, 0),
                -1
            )

        else:

            cv2.circle(
                frame,
                (x, y),
                8,
                (0, 0, 255),
                -1
            )


        # ====================================================
        # HAND SKELETON
        # ====================================================

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


    # ========================================================
    # NO HAND
    # ========================================================

    else:

        previous_x = None
        previous_y = None


    # ========================================================
    # DISPLAY CANVAS
    # ========================================================

    if (
        current_mode != "FREEHAND"
        and shape_preview is not None
    ):

        display_canvas = shape_preview

    else:

        display_canvas = canvas


    # ========================================================
    # COMBINE CAMERA + DRAWING
    # ========================================================

    output = cv2.add(
        frame,
        display_canvas
    )


    # ========================================================
    # BOTTOM STATUS
    # ========================================================

    cv2.putText(
        output,
        "MODE: " + current_mode,
        (10, output.shape[0] - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        output,
        "COLOR:",
        (220, output.shape[0] - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.circle(
        output,
        (300, output.shape[0] - 60),
        12,
        current_color,
        -1
    )


    cv2.putText(
        output,
        "INDEX UP = DRAW | INDEX DOWN = STOP",
        (340, output.shape[0] - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW NORMAL WINDOW
    # ========================================================

    cv2.imshow(
        window_name,
        output
    )


    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Quit
    if key == ord("q"):

        break


    # Clear
    elif key == ord("c"):

        clear_canvas()


    # Undo
    elif key == ord("u"):

        undo()


    # Redo
    elif key == ord("r"):

        redo()


    # Freehand
    elif key == ord("f"):

        current_mode = "FREEHAND"

        print("Mode: FREEHAND")


    # Circle
    elif key == ord("1"):

        current_mode = "CIRCLE"

        print("Mode: CIRCLE")


    # Rectangle
    elif key == ord("2"):

        current_mode = "RECTANGLE"

        print("Mode: RECTANGLE")


    # Triangle
    elif key == ord("3"):

        current_mode = "TRIANGLE"

        print("Mode: TRIANGLE")


    # Cube
    elif key == ord("4"):

        current_mode = "CUBE"

        print("Mode: CUBE")


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()