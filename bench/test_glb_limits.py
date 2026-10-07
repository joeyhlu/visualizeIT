"""Bounded geometry limits: triangle indices differ from vertex counts."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from .glb_model import read_glb


def synthetic_glb(*, index_count=431994, declared_indices=None, declared_vertices=3,
                  nodes=1, index_component=5123, index_kind='SCALAR'):
    vertices = np.asarray([[0, 0, 0, 0, 0, 1, 0, 0],
                           [1, 0, 0, 0, 0, 1, 1, 0],
                           [0, 1, 0, 0, 0, 1, 0, 1]], dtype='<f4')
    dtype = {5123: '<u2', 5125: '<u4', 5126: '<f4'}[index_component]
    values = [0.5, 1.0, 2.0] if index_component == 5126 else [0, 1, 2]
    indices = np.tile(np.asarray(values, dtype=dtype), index_count // 3)
    binary = vertices.tobytes() + indices.tobytes()
    binary += b'\0' * (-len(binary) % 4)
    accessors = [dict(bufferView=0, byteOffset=offset, componentType=5126,
                      count=declared_vertices, type=kind)
                 for offset, kind in ((0, 'VEC3'), (12, 'VEC3'), (24, 'VEC2'))]
    accessors.append(dict(bufferView=1, componentType=index_component,
                          count=index_count if declared_indices is None else declared_indices,
                          type=index_kind))
    document = dict(asset=dict(version='2.0'), buffers=[dict(byteLength=len(binary))],
                    bufferViews=[dict(buffer=0, byteOffset=0, byteLength=96, byteStride=32),
                                 dict(buffer=0, byteOffset=96, byteLength=indices.nbytes)],
                    accessors=accessors,
                    meshes=[dict(primitives=[dict(attributes=dict(POSITION=0, NORMAL=1, TEXCOORD_0=2), indices=3)])],
                    nodes=[dict(mesh=0) for _ in range(nodes)], scenes=[dict(nodes=list(range(nodes)))], scene=0)
    text = json.dumps(document).encode()
    text += b' ' * (-len(text) % 4)
    chunks = struct.pack('<II', len(text), 0x4e4f534a) + text + struct.pack('<II', len(binary), 0x004e4942) + binary
    return struct.pack('<III', 0x46546c67, 2, len(chunks) + 12) + chunks


class GLBGeometryLimitsTests(unittest.TestCase):
    def load(self, raw):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.glb'
            path.write_bytes(raw)
            return read_glb(path, 'synthetic')

    def test_large_index_buffer_still_has_small_bounded_vertex_table(self):
        mesh = self.load(synthetic_glb())
        self.assertEqual(mesh.positions.shape, (3, 3))
        self.assertEqual(mesh.triangles.shape, (143998, 3))
        np.testing.assert_array_equal(mesh.triangles[0], [0, 1, 2])
        np.testing.assert_array_equal(mesh.triangles[-1], [0, 1, 2])

    def test_index_count_over_triangle_budget_rejected(self):
        with self.assertRaisesRegex(ValueError, 'accessor out of bounds'):
            self.load(synthetic_glb(index_count=3, declared_indices=900003))

    def test_vertex_budget_remains_unchanged_and_bool_counts_rejected(self):
        for count in (300001, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'accessor out of bounds'):
                self.load(synthetic_glb(index_count=3, declared_vertices=count))

    def test_repeated_nodes_cannot_multiply_past_scene_triangle_budget(self):
        with self.assertRaisesRegex(ValueError, 'Combined scene geometry'):
            self.load(synthetic_glb(nodes=3))

    def test_float_fractional_or_vector_indices_rejected_before_cast(self):
        for component, kind in ((5126, 'SCALAR'), (5123, 'VEC2'), (5123, 'VEC3')):
            with self.subTest(component=component, kind=kind):
                with self.assertRaisesRegex(ValueError, 'unsigned SCALAR'):
                    self.load(synthetic_glb(index_count=3, index_component=component, index_kind=kind))

    def test_valid_unsigned_32bit_indices_preserve_triangle_values(self):
        mesh = self.load(synthetic_glb(index_count=3, index_component=5125))
        np.testing.assert_array_equal(mesh.triangles, [[0, 1, 2]])
