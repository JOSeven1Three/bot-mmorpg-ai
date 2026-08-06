def test_client_rect_to_screen_rect():
    from bot_mmorpg.scripts.grabscreen import client_rect_to_screen_rect

    rect = client_rect_to_screen_rect((0, 0, 2560, 1009), (120, 48))

    assert rect == (120, 48, 2680, 1057)


def test_find_game_window_tries_diablo_titles(monkeypatch):
    from bot_mmorpg.scripts import grabscreen

    seen = []

    def fake_find_window_region(title, client_area=True):
        seen.append((title, client_area))
        if title == "Diablo IV":
            return (10, 20, 2570, 1029)
        return None

    monkeypatch.setattr(grabscreen, "find_window_region", fake_find_window_region)

    region = grabscreen.find_game_window("diablo_4")

    assert region == (10, 20, 2570, 1029)
    assert seen[0] == ("Diablo IV", True)
