"""Fine mode (SPEC 6.3) at the frame's border: the window hangs over the frame and keeps its size,
what the model finds in the padding is not measured, an object cut by the frame is at the EDGE,
and one that has left the frame is lost while the window stays where it was (review focus 4).

Expected values come from the paths of the scene made here, worked out in its text, from the
ground truth of outline_tracker/synthetic.py (the true mask is the part of the object inside the
frame), and from disks of known size and place (tests/analytic_shapes.py).

Coordinates: px in Tracker's convention (SPEC 3.1): u to the right, v downward, pixel (column c,
row r) has its center at (c + 0.5, r + 0.5). A box is (c0, r0, width, height) in whole px of the
full frame; a fine crop's corner may be negative. Frames are video frame numbers.
"""

import analytic_shapes as shapes
import numpy as np
import pytest
from helpers import mask_in_image
from tracking_helpers import Watched, assert_centered, make_session, run, table_truth, track

from outline_tracker import synthetic
from outline_tracker.measure import mask_center
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.base import MaskResult, crop_to_bbox
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Session
from outline_tracker.synthetic_shapes import Ellipse, Straight
from outline_tracker.tracking_fine import FollowCrop

GRID = list(range(0, 120, 2))  # the clip has 120 frames; the sessions here take every 2nd


@pytest.fixture(scope="module")
def border_clip(tmp_path_factory):
    """Two 47 x 20 px bodies that swim out of a 320 x 240 px frame, 1 px per frame.
    - A through the left border along v = 120.4: its center is at u = 60.3 - frame, its ends at
      u = 36.8 - frame and 83.8 - frame. Column 0 has its pixel centers at u = 0.5, so A is whole up
      to frame 36, cut by the border on frames 37 to 83, and gone from frame 84 on.
    - C through the bottom border along u = 200.4: its center is at v = 180.3 + frame, its ends at
      v = 156.8 + frame and 203.8 + frame. Row 239 has its pixel centers at v = 239.5, so C is whole
      up to frame 35, cut on frames 36 to 82, and gone from frame 83 on."""
    body = Ellipse(23.5, 10.0)
    objects = (synthetic.SceneObject("A", body, Straight((60.3, 120.4), (-1.0, 0.0))),
               synthetic.SceneObject("C", body, Straight((200.4, 180.3), (0.0, 1.0))))
    scene = synthetic.Scene((320, 240), synthetic.FPS, 120, objects, mm_per_px=0.010)
    return synthetic.render(scene, tmp_path_factory.mktemp("border") / "border_tracker.mp4")


# For each body of `border_clip`: the pixels of a mask [row, column] that lie on the border it
# leaves through, and on how many frames of the grid (every 2nd frame) it is whole, cut and gone.
BORDERS = {"A": (np.s_[:, 0], (19, 23, 18)), "C": (np.s_[-1, :], (18, 24, 18))}


def _border_truth(clip, name):
    """Of one body of `border_clip` on the grid: the true centroids and areas (px, of the part
    inside the frame), whether it is seen at all, and whether it has a pixel on its border."""
    border, counts = BORDERS[name]
    true_u, true_v, true_area = table_truth(clip, name, GRID)
    touching = np.array([bool(clip.mask(name, frame)[border].any()) for frame in GRID])
    seen = true_area > 0
    assert ((seen & ~touching).sum(), touching.sum(), (~seen).sum()) == counts  # by the path
    return true_u, true_v, true_area, seen, touching


def _inside_the_frame(arrays, rows):
    """Do the mask crops of the given rows of a track lie inside the 320 x 240 px frame?"""
    offset, shape = arrays.mask_offset[rows], arrays.mask_shape[rows]
    return bool((offset >= 0).all() and (offset[:, 0] + shape[:, 1] <= 320).all()
                and (offset[:, 1] + shape[:, 0] <= 240).all())


@pytest.mark.parametrize("name", ["A", "C"], ids=["left border", "bottom border"])
def test_at_the_frame_border_the_crop_keeps_its_size_and_the_cut_object_is_at_the_edge(border_clip, tmp_path, name):
    session = make_session(border_clip, tmp_path, [track(border_clip, name, mode="fine")])
    segmenter = Watched(ExactFake(border_clip))
    status, seen_by_job = run(border_clip, session, tmp_path, segmenter)
    assert status == "complete" and seen_by_job.finished == ["complete"]
    true_u, true_v, true_area, seen, touching = _border_truth(border_clip, name)

    window = session.tracks[0].fine_window_px
    assert 139 <= window <= 144
    views = segmenter.views[1:]
    # the crop keeps its size, although it hangs over the border by up to half its width
    assert len(segmenter.shapes) == 60 and set(segmenter.shapes) == {(window, window, 3)}
    assert {size for _, _, size in views} == {(window, window)}
    assert max(max(-c0, -r0, c0 + window - 320, r0 + window - 240) for _, (c0, r0), _ in views) > 60

    arrays = ResultsStore.load(tmp_path / "results.npz").arrays(name)
    assert arrays.frames.tolist() == GRID
    assert arrays.visible.tolist() == seen.tolist()
    # The window's own border is far away, in the padding. The mask touches the frame's border: EDGE.
    assert arrays.edge.tolist() == touching.tolist()
    np.testing.assert_allclose(arrays.u[seen], true_u[seen], rtol=0, atol=0.01)
    np.testing.assert_allclose(arrays.v[seen], true_v[seen], rtol=0, atol=0.01)
    assert arrays.area_px.tolist() == true_area.tolist()
    assert _inside_the_frame(arrays, seen)

    # once it has left the frame: lost rows, and the crop stays where it was last seen (frame 82)
    assert np.isnan(arrays.u[~seen]).all() and np.isnan(arrays.v[~seen]).all()
    assert not arrays.edge[~seen].any()
    last = int(np.flatnonzero(seen)[-1])
    assert GRID[last] == 82
    for view in views[last + 1:]:
        assert_centered(view, window, (true_u[last], true_v[last]))
    assert len({offset for _, offset, _ in views[last + 1:]}) == 1
    saved = Session.load(tmp_path / "session.json")
    assert saved.runs[0].frames_done == 60 and saved.complete is True


class Smearing:
    """A stand-in that takes the repeated edge pixels of a padded crop for more of the object, as
    a model may: in the padding it returns the ground truth of the nearest frame pixel, mask and
    logits. The runner tells it the view as it tells `ExactFake` (frame, corner, size)."""

    def __init__(self, clip):
        self.clip, self.view, self.names = clip, None, None

    def set_view(self, frame, offset, size):
        self.view = (frame, offset, size)

    def _results(self, names):
        frame, (c0, r0), (width, height) = self.view
        frame_width, frame_height = self.clip.scene.size
        nearest = np.ix_(np.clip(np.arange(r0, r0 + height), 0, frame_height - 1),
                         np.clip(np.arange(c0, c0 + width), 0, frame_width - 1))
        return [crop_to_bbox(self.clip.mask(name, frame)[nearest], self.clip.logits(name, frame)[nearest],
                             obj_id=name) for name in names]

    def start(self, image, prompts):
        self.names = [prompt.obj_id for prompt in prompts]
        return self._results(self.names)

    def step(self, image):
        return self._results(self.names)

    def preview(self, image, prompts):
        return self._results([prompt.obj_id for prompt in prompts])

    def close(self):
        self.names = None


@pytest.mark.parametrize("name", ["A", "C"], ids=["left border", "bottom border"])
def test_what_the_model_finds_in_the_padding_outside_the_frame_is_not_measured(border_clip, tmp_path, name):
    session = make_session(border_clip, tmp_path, [track(border_clip, name, mode="fine")])
    status, _ = run(border_clip, session, tmp_path, Smearing(border_clip))
    assert status == "complete"
    true_u, true_v, true_area, seen, touching = _border_truth(border_clip, name)

    # Where the body is cut by the frame's border, this stand-in's mask runs on to the end of the
    # window. Only the pixels of the frame count: the true centroid and area of what is in it.
    arrays = ResultsStore.load(tmp_path / "results.npz").arrays(name)
    assert arrays.visible.tolist() == seen.tolist()
    assert arrays.area_px.tolist() == true_area.tolist()
    np.testing.assert_allclose(arrays.u[seen], true_u[seen], rtol=0, atol=0.01)
    np.testing.assert_allclose(arrays.v[seen], true_v[seen], rtol=0, atol=0.01)
    assert arrays.edge.tolist() == touching.tolist()
    assert _inside_the_frame(arrays, seen)
    # the outline closes on the frame's border: u = 0 on the left, v = 240 at the bottom
    assert np.nanmin(arrays.outline_px[:, :, 0]) >= -1e-3 and np.nanmax(arrays.outline_px[:, :, 1]) <= 240 + 1e-3


@pytest.mark.parametrize("disk_at, cut", [
    ((4.3, 40.2), True), ((50.4, 3.6), True), ((96.3, 40.2), True), ((50.4, 76.7), True), ((50.4, 40.2), False),
], ids=["left", "top", "right", "bottom", "inside"])
def test_on_every_side_of_the_frame_only_what_is_in_it_is_measured_and_marked_edge(disk_at, cut):
    # A 100 x 80 px frame and a disk of radius 10 px whose center is 4 px or less inside one of its
    # borders (or in its middle). The 60 px window around the disk hangs over that border; in the
    # padding the model sees the disk go on, and returns all of it.
    frame_size, window = (100, 80), 60
    crop = FollowCrop(window, disk_at, frame_size)
    c0, r0, _, _ = crop.box
    u, v = shapes.pixel_centers(window, window, origin=(c0, r0))
    found = shapes.to_result(shapes.disk(u, v, disk_at, 10.0))  # offset in px of the crop
    record = crop.measure(found, 7, 0.1)

    # the truth: the disk's pixels in the frame
    u, v = shapes.pixel_centers(80, 100)
    in_frame = shapes.disk(u, v, disk_at, 10.0) > 0
    true_u, true_v, true_area = mask_center(in_frame)
    assert (true_area < 300) is cut  # pi 10^2 = 314 px when it is whole
    assert record.frame == 7 and record.visible and record.mode == "fine"
    assert record.area_px == true_area
    assert (record.u, record.v) == pytest.approx((true_u, true_v), abs=1e-9)
    assert record.edge is cut
    assert record.cell_px == window / 256
    np.testing.assert_array_equal(mask_in_image(MaskResult("A", record.mask_offset, record.mask(), None, None),
                                                (80, 100)), in_frame)
    assert crop.center == (record.u, record.v)  # where the next frame's window will be
