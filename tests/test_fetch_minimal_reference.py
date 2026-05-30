import json
import struct
from unittest.mock import MagicMock, patch

import pytest

from fetch_minimal_reference import (
    fetch_model_shapes,
    fetch_shard_header,
)


def make_shard_bytes(tensors: dict) -> tuple[bytes, bytes]:
    """Return (size_bytes, header_bytes) for a fake safetensors shard."""
    header_json = json.dumps(tensors).encode()
    size_bytes = struct.pack("<Q", len(header_json))
    return size_bytes, header_json


class TestFetchShardHeader:
    def _make_session(self, tensors):
        size_bytes, header_bytes = make_shard_bytes(tensors)
        size_resp = MagicMock(content=size_bytes, raise_for_status=MagicMock())
        header_resp = MagicMock(
            content=header_bytes, raise_for_status=MagicMock()
        )
        session = MagicMock()
        session.get.side_effect = [size_resp, header_resp]
        return session

    def test_parses_tensors(self):
        tensors = {
            "model.embed_tokens.weight": {
                "shape": [32000, 4096],
                "dtype": "BF16",
                "data_offsets": [0, 100],
            },
            "model.layers.0.self_attn.q_proj.weight": {
                "shape": [4096, 4096],
                "dtype": "BF16",
                "data_offsets": [100, 200],
            },
        }
        result = fetch_shard_header(
            self._make_session(tensors), "http://example.com/model.safetensors"
        )

        assert result == {
            "model.embed_tokens.weight": {
                "shape": [32000, 4096],
                "dtype": "BF16",
            },
            "model.layers.0.self_attn.q_proj.weight": {
                "shape": [4096, 4096],
                "dtype": "BF16",
            },
        }

    def test_skips_metadata_key(self):
        tensors = {
            "__metadata__": {"format": "pt"},
            "model.embed_tokens.weight": {
                "shape": [32000, 4096],
                "dtype": "BF16",
                "data_offsets": [0, 100],
            },
        }
        result = fetch_shard_header(
            self._make_session(tensors), "http://example.com/model.safetensors"
        )

        assert "__metadata__" not in result
        assert len(result) == 1

    def test_exits_on_http_error(self):
        session = MagicMock()
        session.get.return_value.ok = False
        session.get.return_value.status_code = 404
        session.get.return_value.reason = "Not Found"

        with pytest.raises(SystemExit):
            fetch_shard_header(
                session, "http://example.com/missing.safetensors"
            )


class TestFetchModelShapes:
    def _shard_responses(self, tensors):
        size_bytes, header_bytes = make_shard_bytes(tensors)
        return [
            MagicMock(content=size_bytes, raise_for_status=MagicMock()),
            MagicMock(content=header_bytes, raise_for_status=MagicMock()),
        ]

    def _patch_session(self, mock_session_cls, side_effect):
        session = mock_session_cls.return_value.__enter__.return_value
        session.get.side_effect = side_effect
        return session

    def test_multi_shard_model(self):
        index = {
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            }
        }
        shard1_tensors = {
            "model.embed_tokens.weight": {
                "shape": [32000, 4096],
                "dtype": "BF16",
                "data_offsets": [0, 100],
            }
        }
        shard2_tensors = {
            "model.layers.0.self_attn.q_proj.weight": {
                "shape": [4096, 4096],
                "dtype": "BF16",
                "data_offsets": [0, 100],
            }
        }

        index_resp = MagicMock(
            status_code=200, json=MagicMock(return_value=index)
        )

        with patch(
            "fetch_minimal_reference.requests.Session"
        ) as mock_session_cls:
            self._patch_session(
                mock_session_cls,
                (
                    [
                        index_resp,
                        *self._shard_responses(shard1_tensors),
                        *self._shard_responses(shard2_tensors),
                    ]
                ),
            )
            result = fetch_model_shapes(
                "org/model", "main", "https://huggingface.co"
            )

        assert "model.embed_tokens.weight" in result
        assert "model.layers.0.self_attn.q_proj.weight" in result

    def test_single_shard_model(self):
        tensors = {
            "model.embed_tokens.weight": {
                "shape": [32000, 4096],
                "dtype": "BF16",
                "data_offsets": [0, 100],
            }
        }
        index_resp = MagicMock(status_code=404)

        with patch(
            "fetch_minimal_reference.requests.Session"
        ) as mock_session_cls:
            self._patch_session(
                mock_session_cls, [index_resp, *self._shard_responses(tensors)]
            )
            result = fetch_model_shapes(
                "org/single-shard-model", "main", "https://huggingface.co"
            )

        assert "model.embed_tokens.weight" in result

    def test_unexpected_http_status_exits(self):
        index_resp = MagicMock(status_code=500, reason="Internal Server Error")

        with patch(
            "fetch_minimal_reference.requests.Session"
        ) as mock_session_cls:
            self._patch_session(mock_session_cls, [index_resp])

            with pytest.raises(SystemExit):
                fetch_model_shapes(
                    "org/model", "main", "https://huggingface.co"
                )
