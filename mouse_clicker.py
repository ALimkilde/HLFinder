import time
import pyautogui

# -----------------------
# CONFIGURATION
# -----------------------
X_MINUTES = 40           # total runtime in minutes
INTERVAL = 100            # interval in seconds
MOVE_CM = 0.6585         # Seems good for 67% ; how far to move the mouse down
DELAY = 5                # Delay in sec
MAX_ITER = 34            # Maximum iterations
# -----------------------

# Approximate pixel conversion (38 px ≈ 1 cm on 96 DPI screens)
PIXELS_PER_CM = 38
MOVE_PIXELS = int(MOVE_CM * PIXELS_PER_CM)

end_time = time.time() + X_MINUTES * 60

for i in range(DELAY):
    print(f"Starting in {DELAY - i} seconds\r",end="")
    time.sleep(1)

print(f"Running for {X_MINUTES} minutes...")

iter = 0
while (iter<MAX_ITER and time.time() < end_time):

    # Click
    pyautogui.click()

    # Move mouse down by 0.9 cm (≈34 pixels)
    x, y = pyautogui.position()
    pyautogui.moveTo(x, y + MOVE_PIXELS)

    iter += 1
    print(f"Iter {iter} of {MAX_ITER}: Moved + clicked\r",end="")

    # Countdown wait
    for remaining in range(INTERVAL, 0, -1):
        print(f"Waiting: {remaining} s\r", end="", flush=True)
        time.sleep(1)

    print(" " * 30 + "\r", end="")  # clear the line

print("Done.")

