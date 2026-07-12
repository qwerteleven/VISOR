import numpy as np
import pytest
import torch
from PIL import Image


from utils.image_processing import draw_user_rectangle, overlay_masks


def make_frame(width=100, height=50, color=(0, 0, 0)):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def test_draw_user_rectangle_requires_exactly_two_points():
    frame = make_frame()
    config = {"color": (255, 0, 0), "thickness": 1}

    with pytest.raises(AssertionError):
        draw_user_rectangle(frame, config, [[0.1, 0.1]])

    with pytest.raises(AssertionError):
        draw_user_rectangle(
            frame, config, [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]
        )  # three points


def test_draw_user_rectangle_converts_normalized_coordinates_correctly():
    """

    user_ref_point is normalized [0,1]; verify it maps to the expected
    pixel coordinates for a known frame size, by checking the corner
    pixels of the drawn rectangle land where math predicts.

    """
    width, height = 100, 50
    frame = make_frame(width, height, color=(0, 0, 0))
    config = {"color": (255, 255, 255), "thickness": 1}

    user_ref_point = [[0.2, 0.4], [0.8, 0.6]]
    expected_a = (int(0.2 * width), int(0.4 * height))
    expected_b = (int(0.8 * width), int(0.6 * height))

    result = draw_user_rectangle(frame, config, user_ref_point)

    assert tuple(result[expected_a[1], expected_a[0]]) == config["color"]
    assert tuple(result[expected_b[1], expected_b[0]]) == config["color"]


def test_draw_user_rectangle_full_frame_bounds():
    width, height = 40, 20
    frame = make_frame(width, height)
    config = {"color": (0, 255, 0), "thickness": 1}

    result = draw_user_rectangle(frame, config, [[0.0, 0.0], [1.0, 1.0]])

    assert tuple(result[0, 0]) == config["color"]
    assert tuple(result[height - 1, 0]) == config["color"]


def test_draw_user_rectangle_mutates_input_array_in_place():
    frame = make_frame(20, 20)
    config = {"color": (255, 0, 0), "thickness": 1}

    result = draw_user_rectangle(frame, config, [[0.1, 0.1], [0.9, 0.9]])

    assert result is frame
    assert np.array_equal(result, frame)


def test_draw_user_rectangle_respects_thickness():
    width, height = 60, 60
    user_ref_point = [[0.2, 0.2], [0.8, 0.8]]

    thin_frame = make_frame(width, height)
    thin_result = draw_user_rectangle(
        thin_frame, {"color": (255, 0, 0), "thickness": 1}, user_ref_point
    )
    thin_painted = np.count_nonzero(np.all(thin_result == (255, 0, 0), axis=-1))

    thick_frame = make_frame(width, height)
    thick_result = draw_user_rectangle(
        thick_frame, {"color": (255, 0, 0), "thickness": 5}, user_ref_point
    )
    thick_painted = np.count_nonzero(np.all(thick_result == (255, 0, 0), axis=-1))

    assert thick_painted > thin_painted


def test_draw_user_rectangle_does_not_paint_outside_the_box_interior():
    """
    A thin rectangle should leave the interior untouched — only the
    border is drawn, not a filled rectangle.
    """

    width, height = 60, 60
    frame = make_frame(width, height, color=(0, 0, 0))
    config = {"color": (255, 0, 0), "thickness": 1}

    result = draw_user_rectangle(frame, config, [[0.1, 0.1], [0.9, 0.9]])

    center_pixel = result[height // 2, width // 2]
    assert tuple(center_pixel) == (0, 0, 0)


def make_rgb_image(width=8, height=8, color=(10, 20, 30)):
    return Image.new("RGB", (width, height), color)


def first_rainbow_color(n_masks=1):
    import matplotlib

    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    return tuple(int(c * 255) for c in cmap(0)[:3])


def test_overlay_masks_requires_pil_image_not_numpy_array():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    masks = torch.zeros((1, 8, 8))

    with pytest.raises(AttributeError):
        overlay_masks(image, masks)


def test_overlay_masks_returns_rgba_image_of_same_size():
    image = make_rgb_image(8, 8)
    masks = torch.zeros((1, 8, 8))

    result = overlay_masks(image, masks)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGBA"
    assert result.size == image.size


def test_overlay_masks_with_zero_masks_returns_image_unchanged_besides_mode():
    image = make_rgb_image(8, 8, color=(50, 60, 70))
    masks = torch.zeros((0, 8, 8))

    result = overlay_masks(image, masks)

    expected = image.convert("RGBA")
    assert np.array_equal(np.array(result), np.array(expected))


def test_overlay_masks_all_zero_mask_leaves_pixels_visually_unchanged():
    image = make_rgb_image(8, 8, color=(50, 60, 70))
    masks = torch.zeros((1, 8, 8))

    result = overlay_masks(image, masks)

    expected_rgba = image.convert("RGBA")
    result_arr = np.array(result)[:, :, :3]
    expected_arr = np.array(expected_rgba)[:, :, :3]
    assert np.array_equal(result_arr, expected_arr)


def test_overlay_masks_all_one_mask_blends_to_expected_color():
    bg_color = (50, 60, 70)
    image = make_rgb_image(4, 4, color=bg_color)
    masks = torch.ones((1, 4, 4))

    result = overlay_masks(image, masks)

    fg_color = first_rainbow_color(n_masks=1)
    alpha = 127 / 255

    expected_rgb = tuple(
        round(fg_color[c] * alpha + bg_color[c] * (1 - alpha)) for c in range(3)
    )

    got = result.getpixel((0, 0))[:3]

    assert all(abs(got[c] - expected_rgb[c]) <= 2 for c in range(3)), (
        got,
        expected_rgb,
    )


def test_overlay_masks_soft_probability_values_are_silently_treated_as_zero():
    image = make_rgb_image(4, 4, color=(50, 60, 70))
    soft_mask = torch.full((1, 4, 4), 0.7)

    result = overlay_masks(image, soft_mask)

    expected_rgba = image.convert("RGBA")
    result_arr = np.array(result)[:, :, :3]
    expected_arr = np.array(expected_rgba)[:, :, :3]
    assert np.array_equal(result_arr, expected_arr)


def test_overlay_masks_partial_region_only_tints_masked_pixels():
    image = make_rgb_image(4, 4, color=(50, 60, 70))
    mask = torch.zeros((1, 4, 4))
    mask[0, :, :2] = 1.0

    result = overlay_masks(image, mask)

    left_pixel = result.getpixel((0, 0))[:3]
    right_pixel = result.getpixel((3, 0))[:3]

    assert left_pixel != (50, 60, 70)
    assert right_pixel == (50, 60, 70)


def test_overlay_masks_multiple_masks_use_distinct_colors():
    image = make_rgb_image(4, 4, color=(0, 0, 0))
    masks = torch.zeros((2, 4, 4))
    masks[0, :, :2] = 1.0
    masks[1, :, 2:] = 1.0

    result = overlay_masks(image, masks)

    left_pixel = result.getpixel((0, 0))[:3]
    right_pixel = result.getpixel((3, 0))[:3]

    assert left_pixel != right_pixel
    assert left_pixel != (0, 0, 0)
    assert right_pixel != (0, 0, 0)
