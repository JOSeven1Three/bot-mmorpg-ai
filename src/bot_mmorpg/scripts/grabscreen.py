# Done by Frannecklp
# Cross-platform support added for Linux/macOS

import base64
import platform

import cv2
import numpy as np

# Platform detection for cross-platform support
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    try:
        import win32api
        import win32con
        import win32gui
        import win32ui

        _WIN32_AVAILABLE = True
    except ImportError:
        _WIN32_AVAILABLE = False
else:
    _WIN32_AVAILABLE = False

# Cross-platform fallback using mss (works on all platforms)
try:
    import mss

    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False


def list_monitors():
    """
    List all available monitors/screens.

    Returns:
        list: List of monitor dictionaries with id, name, and dimensions.
              Example: [{"id": 0, "name": "Primary (1920x1080)", "width": 1920, "height": 1080, "left": 0, "top": 0}]
    """
    monitors = []

    if _MSS_AVAILABLE:
        with mss.mss() as sct:
            # Skip monitor 0 (it's the combined virtual screen on Windows)
            for i, mon in enumerate(sct.monitors[1:], start=1):
                monitors.append(
                    {
                        "id": i,
                        "name": f"Monitor {i} ({mon['width']}x{mon['height']})",
                        "width": mon["width"],
                        "height": mon["height"],
                        "left": mon["left"],
                        "top": mon["top"],
                    }
                )
    elif IS_WINDOWS and _WIN32_AVAILABLE:
        # Fallback: just report primary monitor
        width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        monitors.append(
            {
                "id": 1,
                "name": f"Primary ({width}x{height})",
                "width": width,
                "height": height,
                "left": 0,
                "top": 0,
            }
        )
    else:
        # Default fallback
        monitors.append(
            {
                "id": 1,
                "name": "Primary (Unknown)",
                "width": 1920,
                "height": 1080,
                "left": 0,
                "top": 0,
            }
        )

    return monitors


def grab_screen_monitor(monitor_id=1):
    """
    Capture a specific monitor by ID.

    Args:
        monitor_id: Monitor ID (1-based, from list_monitors())

    Returns:
        numpy.ndarray: RGB image array of the captured monitor.
    """
    if _MSS_AVAILABLE:
        with mss.mss() as sct:
            if monitor_id < 1 or monitor_id >= len(sct.monitors):
                monitor_id = 1  # Default to primary
            monitor = sct.monitors[monitor_id]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # Fallback to full screen
    return grab_screen()


def grab_screen_thumbnail(monitor_id=1, max_width=320, max_height=180):
    """
    Capture a monitor and return a thumbnail-sized image.
    Useful for preview displays.

    Args:
        monitor_id: Monitor ID to capture
        max_width: Maximum thumbnail width
        max_height: Maximum thumbnail height

    Returns:
        numpy.ndarray: Resized RGB image array
    """
    img = grab_screen_monitor(monitor_id)
    h, w = img.shape[:2]

    # Calculate scale to fit within max dimensions
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def grab_screen_base64(monitor_id=1, max_width=640, max_height=360, quality=70):
    """
    Capture a monitor and return as base64 JPEG string.
    Perfect for sending to web UI.

    Args:
        monitor_id: Monitor ID to capture
        max_width: Maximum image width
        max_height: Maximum image height
        quality: JPEG quality (1-100)

    Returns:
        str: Base64-encoded JPEG image data
    """
    img = grab_screen_thumbnail(monitor_id, max_width, max_height)
    # Convert RGB to BGR for OpenCV encoding
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Encode as JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode(".jpg", img_bgr, encode_param)
    # Convert to base64
    return base64.b64encode(buffer).decode("utf-8")


def _grab_screen_win32(region=None):
    """Windows-specific screen capture using Win32 API (original implementation)."""
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    # Fix: Use np.frombuffer instead of deprecated np.fromstring
    img = np.frombuffer(signedIntsArray, dtype="uint8")
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def _grab_screen_mss(region=None):
    """Cross-platform screen capture using mss library."""
    with mss.mss() as sct:
        if region:
            left, top, x2, y2 = region
            monitor = {
                "left": left,
                "top": top,
                "width": x2 - left + 1,
                "height": y2 - top + 1,
            }
        else:
            # Capture full screen (primary monitor)
            monitor = sct.monitors[1]

        screenshot = sct.grab(monitor)
        # Convert to numpy array
        img = np.array(screenshot)
        # mss returns BGRA, convert to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def client_rect_to_screen_rect(client_rect, client_origin):
    """
    Convert a window client rect plus screen-space origin into screen bounds.

    Args:
        client_rect: Tuple (left, top, right, bottom) in client coordinates.
        client_origin: Tuple (x, y) for the client area's top-left on screen.

    Returns:
        tuple: (left, top, right, bottom) in screen coordinates.
    """
    left, top, right, bottom = client_rect
    origin_x, origin_y = client_origin
    width = right - left
    height = bottom - top
    return (origin_x, origin_y, origin_x + width, origin_y + height)


def get_window_region(hwnd, client_area=True):
    """
    Resolve a window's screen region.

    Args:
        hwnd: Native window handle
        client_area: If True, return the exact client rectangle.
                     If False, return the outer window rectangle.

    Returns:
        tuple: (left, top, right, bottom) screen coordinates.
    """
    if not (IS_WINDOWS and _WIN32_AVAILABLE):
        return None

    if not client_area:
        return win32gui.GetWindowRect(hwnd)

    client_rect = win32gui.GetClientRect(hwnd)
    origin = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))
    return client_rect_to_screen_rect(client_rect, origin)


def find_window_region(window_title, client_area=True):
    """
    Find a game window by title and return its screen region.

    Searches for windows whose title contains the given string
    (case-insensitive). Useful for auto-detecting game window position
    instead of requiring manual --region coordinates.

    Args:
        window_title: Partial or full window title to search for.

    Returns:
        tuple: (left, top, right, bottom) pixel coordinates, or None if not found.
    """
    if IS_WINDOWS and _WIN32_AVAILABLE:
        result = []

        def _enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if window_title.lower() in title.lower():
                    rect = get_window_region(hwnd, client_area=client_area)
                    result.append(rect)

        win32gui.EnumWindows(_enum_callback, None)
        if result:
            return result[0]
        return None

    # On Linux, attempt wmctrl-style detection via subprocess
    try:
        import subprocess

        output = subprocess.check_output(
            ["wmctrl", "-lG"], stderr=subprocess.DEVNULL, text=True
        )
        for line in output.strip().split("\n"):
            if window_title.lower() in line.lower():
                parts = line.split()
                # wmctrl -lG format: id desktop x y w h hostname title...
                if len(parts) >= 7:
                    x, y, w, h = (
                        int(parts[2]),
                        int(parts[3]),
                        int(parts[4]),
                        int(parts[5]),
                    )
                    return (x, y, x + w, y + h)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return None


# Common window titles for supported games (used by --game auto-detect)
GAME_WINDOW_TITLES = {
    "dragon_ball_online": [
        "Dragon Ball Online",
        "DragonBall Online",
        "DBO",
        "DBOG",
        "Ultimate DBO",
        "UltimateDBO",
    ],
    "genshin_impact": ["Genshin Impact", "GenshinImpact", "原神"],
    "world_of_warcraft": ["World of Warcraft", "WoW", "Wow-64"],
    "final_fantasy_xiv": ["FINAL FANTASY XIV", "FFXIV"],
    "guild_wars_2": ["Guild Wars 2"],
    "lost_ark": ["LOST ARK", "Lost Ark"],
    "diablo_4": ["Diablo IV", "Diablo 4", "Diablo® IV"],
    "new_world": ["New World"],
}


def find_game_window(game_id):
    """
    Auto-detect a game window region by game profile ID.

    Tries multiple known window titles for the given game.

    Args:
        game_id: Game identifier from game_profiles (e.g., "dragon_ball_online")

    Returns:
        tuple: (left, top, right, bottom) pixel coordinates, or None if not found.
    """
    titles = GAME_WINDOW_TITLES.get(game_id, [])
    for title in titles:
        region = find_window_region(title)
        if region is not None:
            return region
    return None


def grab_screen(region=None):
    """
    Capture a region of the screen.

    Args:
        region: Optional tuple (left, top, right, bottom) defining the capture area.
                If None, captures the entire screen.

    Returns:
        numpy.ndarray: RGB image array of the captured screen region.

    Cross-platform support:
        - Windows: Uses Win32 API (fast, original implementation)
        - Linux/macOS: Uses mss library (requires: pip install mss)
    """
    # Prefer Win32 on Windows for best performance
    if IS_WINDOWS and _WIN32_AVAILABLE:
        return _grab_screen_win32(region)

    # Fallback to mss (cross-platform)
    if _MSS_AVAILABLE:
        return _grab_screen_mss(region)

    # No screen capture available
    raise RuntimeError(
        "No screen capture method available. "
        "On Windows, install pywin32: pip install pywin32. "
        "On Linux/macOS, install mss: pip install mss"
    )
