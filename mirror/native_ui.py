import cv2

from mirror.logger import logger

viewport_image = "Mirror Mirror"


esc = 27
q = ord("q")
f = ord("f")


def toggle_fullscreen():
    fullscreen = cv2.getWindowProperty(viewport_image, cv2.WND_PROP_FULLSCREEN)
    toggled = int(not (bool(int(fullscreen))))
    logger.info("Fullscreen toggled: %s", toggled)
    cv2.setWindowProperty(viewport_image, cv2.WND_PROP_FULLSCREEN, toggled)


def handle_keystrokes() -> bool:
    key = cv2.pollKey()
    if key in (q, esc):
        return False

    if key == f:
        toggle_fullscreen()
    return True
