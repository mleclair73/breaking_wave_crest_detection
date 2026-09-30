"""Regression tests for strict timestamps, normalization, and tiled inference."""

from datetime import datetime
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import tiling
from models.segnext import SegNeXt
from tiling import predict_tiled, predict_tiled_many
from common.datetime_utils import (
    get_time_resolution_for_date,
    parse_datetime_from_filename,
)
from common.normalization import denorm, renorm
from calculate_wave_speeds import (
    load_prediction_video,
    process_video as process_speed_video,
)
from segmentation.predict_video import (
    DensityMapAccumulator,
    StreamingVideoWriter,
    _validate_output_frame_counts,
    load_video_chunked,
    load_model,
    process_video as process_prediction_video,
    temporal_blend_weights,
)


class _FakeCapture:
    def __init__(self, frames, reported_count, *, read_error_at=None):
        self.frames = frames
        self.reported_count = reported_count
        self.read_error_at = read_error_at
        self.index = 0

    def isOpened(self):
        return True

    def get(self, prop):
        values = {
            cv2.CAP_PROP_FPS: 2.0,
            cv2.CAP_PROP_FRAME_COUNT: self.reported_count,
            cv2.CAP_PROP_FRAME_WIDTH: self.frames[0].shape[1],
            cv2.CAP_PROP_FRAME_HEIGHT: self.frames[0].shape[0],
        }
        return values[prop]

    def read(self):
        if self.index == self.read_error_at:
            raise OSError("synthetic decode failure")
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def release(self):
        pass


def _capture_pair(frames, reported_count, *, read_error_at=None):
    captures = iter(
        (
            _FakeCapture(frames, reported_count),
            _FakeCapture(
                frames,
                reported_count,
                read_error_at=read_error_at,
            ),
        )
    )
    return lambda _path: next(captures)


class TimestampTests(unittest.TestCase):
    def test_argus_timestamp_is_required(self):
        with self.assertRaisesRegex(ValueError, "YYYYmmddTHHMMSSZ"):
            parse_datetime_from_filename("prediction.avi")

    def test_frame_rate_uses_capture_date(self):
        september_19 = parse_datetime_from_filename(
            "ArgusFF_20210919T163000Z_pred.avi"
        )
        september_20 = datetime(2021, 9, 20, 16, 30)
        self.assertEqual(get_time_resolution_for_date(september_19), (1.0, 1.0))
        self.assertEqual(get_time_resolution_for_date(september_20), (0.5, 2.0))

    @patch("calculate_wave_speeds.load_prediction_video")
    def test_wave_speed_pipeline_fails_before_video_io(self, load_video):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "YYYYmmddTHHMMSSZ"):
                process_speed_video(
                    Path("prediction.avi"),
                    Path(directory),
                    n_workers=1,
                )
        load_video.assert_not_called()


class NormalizationTests(unittest.TestCase):
    def test_tensor_round_trip(self):
        image = torch.rand(3, 7, 9)
        torch.testing.assert_close(denorm(renorm(image)), image)


class VideoChunkTests(unittest.TestCase):
    @staticmethod
    def frames(count):
        return [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(count)]

    def test_exact_chunk_boundary_marks_last_chunk(self):
        frames = self.frames(6)
        with patch(
            "segmentation.predict_video.cv2.VideoCapture",
            side_effect=_capture_pair(frames, reported_count=6),
        ):
            chunks = list(
                load_video_chunked(
                    "video.avi",
                    chunk_size=4,
                    overlap_frames=2,
                )
            )

        self.assertEqual(
            [(start, len(chunk), is_last) for chunk, start, _, is_last, _ in chunks],
            [(0, 4, False), (2, 4, True)],
        )
        last_chunk, _, _, is_last, _ = chunks[-1]
        weights = temporal_blend_weights(
            len(last_chunk),
            2,
            fade_in=True,
            fade_out=not is_last,
        )
        self.assertGreater(weights[-1], 0)

    def test_chunk_coverage_for_every_tail_length(self):
        for overlap in range(4):
            for count in range(1, 13):
                with self.subTest(overlap=overlap, count=count):
                    frames = self.frames(count)
                    with patch(
                        "segmentation.predict_video.cv2.VideoCapture",
                        side_effect=_capture_pair(frames, reported_count=count),
                    ):
                        chunks = list(
                            load_video_chunked(
                                "video.avi",
                                chunk_size=4,
                                overlap_frames=overlap,
                            )
                        )
                    self.assertEqual(sum(chunk[3] for chunk in chunks), 1)
                    covered = {
                        start + offset
                        for chunk, start, _, _, _ in chunks
                        for offset in range(len(chunk))
                    }
                    self.assertEqual(covered, set(range(count)))
                    weight_sum = np.zeros(count, dtype=np.float32)
                    for chunk, start, _, is_last, _ in chunks:
                        weights = temporal_blend_weights(
                            len(chunk),
                            overlap,
                            fade_in=start > 0,
                            fade_out=not is_last,
                        )
                        weight_sum[start : start + len(chunk)] += weights
                    self.assertTrue(np.all(weight_sum > 0))

    def test_short_decode_is_rejected(self):
        frames = self.frames(5)
        with patch(
            "segmentation.predict_video.cv2.VideoCapture",
            side_effect=_capture_pair(frames, reported_count=6),
        ):
            with self.assertRaisesRegex(RuntimeError, "Decoded 5/6 expected"):
                list(load_video_chunked("video.avi", chunk_size=4, overlap_frames=2))

    def test_extra_decode_is_rejected(self):
        frames = self.frames(7)
        with patch(
            "segmentation.predict_video.cv2.VideoCapture",
            side_effect=_capture_pair(frames, reported_count=6),
        ):
            with self.assertRaisesRegex(RuntimeError, "additional decodable frames"):
                list(load_video_chunked("video.avi", chunk_size=4, overlap_frames=2))

    def test_explicit_frame_limit_allows_remaining_source_frames(self):
        frames = self.frames(7)
        with patch(
            "segmentation.predict_video.cv2.VideoCapture",
            side_effect=_capture_pair(frames, reported_count=7),
        ):
            chunks = list(
                load_video_chunked(
                    "video.avi",
                    chunk_size=4,
                    overlap_frames=2,
                    max_frames=5,
                )
            )
        self.assertTrue(chunks[-1][3])
        self.assertTrue(all(chunk[4] == 5 for chunk in chunks))

    def test_reader_exception_reaches_consumer(self):
        frames = self.frames(6)
        with patch(
            "segmentation.predict_video.cv2.VideoCapture",
            side_effect=_capture_pair(
                frames,
                reported_count=6,
                read_error_at=3,
            ),
        ):
            with self.assertRaisesRegex(OSError, "synthetic decode failure"):
                list(load_video_chunked("video.avi", chunk_size=4, overlap_frames=2))

    def test_output_counts_must_match_input(self):
        writer = SimpleNamespace(frames_written=6)
        density = SimpleNamespace(total_frames=5)
        with self.assertRaisesRegex(RuntimeError, "density=5"):
            _validate_output_frame_counts(6, 6, writer, writer, density)

    def test_wave_speed_loader_rejects_short_decode(self):
        frames = self.frames(5)
        capture = _FakeCapture(frames, reported_count=6)
        with patch(
            "calculate_wave_speeds.cv2.VideoCapture",
            return_value=capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "Decoded 5/6 expected"):
                load_prediction_video("prediction.avi")

    def test_wave_speed_loader_rejects_extra_decode(self):
        frames = self.frames(7)
        capture = _FakeCapture(frames, reported_count=6)
        with patch(
            "calculate_wave_speeds.cv2.VideoCapture",
            return_value=capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "additional decodable frames"):
                load_prediction_video("prediction.avi")


class TilingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1).eval()
        self.images = np.random.default_rng(11).random(
            (3, 35, 29, 3), dtype=np.float32
        )

    def test_batched_matches_separate_images(self):
        options = {
            "patch_size": 24,
            "overlap_y": 7,
            "overlap_x": 5,
            "batch_size": 4,
            "device": "cpu",
        }
        batched = predict_tiled_many(self.model, self.images, **options)
        separate = np.stack(
            [predict_tiled(self.model, image, **options) for image in self.images]
        )
        np.testing.assert_array_equal(batched, separate)

    def test_axis_overlaps_are_required(self):
        with self.assertRaises(TypeError):
            predict_tiled(self.model, self.images[0], patch_size=24, device="cpu")

    def test_zero_tiles_skip_model_execution(self):
        class FailingModel(torch.nn.Module):
            def forward(self, _inputs):
                raise AssertionError("zero tile reached the model")

        result = predict_tiled(
            FailingModel().eval(),
            np.zeros((24, 24, 3), dtype=np.float32),
            patch_size=24,
            overlap_y=0,
            overlap_x=0,
            device="cpu",
            skip_zero_patches=True,
        )
        np.testing.assert_array_equal(result, np.zeros((24, 24), dtype=np.float32))

    def test_denominator_cache_is_bounded(self):
        tiling._BLEND_DENOMINATOR_CACHE.clear()
        for size in range(2, 24):
            tiling._cached_blend_tensors(
                patch_size=2,
                padded_height=size,
                padded_width=size,
                positions=((0, 0),),
                use_hanning=False,
                device="cpu",
            )
        self.assertLessEqual(
            len(tiling._BLEND_DENOMINATOR_CACHE),
            tiling._MAX_DENOMINATOR_CACHE_ENTRIES,
        )


class PredictionApiTests(unittest.TestCase):
    def test_removed_model_type_and_overlap_parameters_stay_removed(self):
        self.assertNotIn("model_type", inspect.signature(load_model).parameters)
        parameters = inspect.signature(process_prediction_video).parameters
        self.assertNotIn("model_type", parameters)
        self.assertNotIn("overlap", parameters)

    def test_thresholds_are_required(self):
        required = inspect.Parameter.empty
        self.assertIs(
            inspect.signature(process_prediction_video)
            .parameters["detection_threshold"]
            .default,
            required,
        )
        self.assertIs(
            inspect.signature(StreamingVideoWriter.write_prediction_chunk)
            .parameters["threshold"]
            .default,
            required,
        )
        self.assertIs(
            inspect.signature(StreamingVideoWriter.write_overlay_chunk)
            .parameters["threshold"]
            .default,
            required,
        )
        self.assertIs(
            inspect.signature(DensityMapAccumulator.add_chunk)
            .parameters["threshold"]
            .default,
            required,
        )

    def test_production_checkpoint_seed_is_required(self):
        payload = {
            "config": {
                "id": "segnext_t_learned_up_skip_4_2",
                "arch": "segnext",
                "decoder_type": "learned_up",
                "decoder_skips": [4, 2],
            },
            "model_state_dict": {},
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "must contain seed"):
                load_model(checkpoint)


class PretrainedBackboneTests(unittest.TestCase):
    def test_incomplete_pretrained_backbone_is_rejected(self):
        class Holder(torch.nn.Module):
            load_pretrained_backbone = SegNeXt.load_pretrained_backbone

            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(3, 2)

        payload = {"state_dict": {"weight": torch.ones(2, 3)}}
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pretrained.pth"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                Holder().load_pretrained_backbone(checkpoint)


if __name__ == "__main__":
    unittest.main()
