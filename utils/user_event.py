import cv2
import utils.globals as globals


def click_and_crop(event: int, x: int, y: int, flags: int, param) -> None:
    """

        interruption handle of user interection with openCV window

    Args:
        event (int): type of user action
        x (int): x position
        y (int): y position
        flags (int): unnused
        param (int): unnused
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        windowWidth = cv2.getWindowImageRect(globals.VISOR_NAME)[2]
        windowHeight = cv2.getWindowImageRect(globals.VISOR_NAME)[3]
        ref_point = [(x / windowWidth, y / windowHeight)]
        globals.user_ref_point = ref_point

    if event == cv2.EVENT_LBUTTONUP:
        windowWidth = cv2.getWindowImageRect(globals.VISOR_NAME)[2]
        windowHeight = cv2.getWindowImageRect(globals.VISOR_NAME)[3]
        globals.user_ref_point += [(x / windowWidth, y / windowHeight)]

    if event == cv2.EVENT_RBUTTONDOWN:
        windowWidth = cv2.getWindowImageRect(globals.VISOR_NAME)[2]
        windowHeight = cv2.getWindowImageRect(globals.VISOR_NAME)[3]
        ref_point = [(x / windowWidth, y / windowHeight)]
        globals.user_ref_point = ref_point
