import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from mocap.retarget import solver
from mocap.retarget.mapping import PMX, body_mapping


class PmxRetargetTests(unittest.TestCase):
    def test_middle_torso_alias_requires_a_matching_hierarchy(self):
        for alias in ("上半身１", "上半身3", "上半身３"):
            with self.subTest(alias=alias):
                bones = set(PMX) - {"上半身1"} | {alias}
                parents = {"上半身2": alias, alias: "上半身"}
                self.assertEqual(body_mapping("pmx", bones, {}, parents=parents)[6], alias)
                parents[alias] = "腰"
                with self.assertRaisesRegex(ValueError, "Spine1"):
                    body_mapping("pmx", bones, {}, parents=parents)

    def test_captured_pelvis_rotation_avoids_lifting_root_for_pmx_sibling_pivots(self):
        source_rest = np.zeros((22, 3))
        source_rest[3, 1] = 0.10
        target_rest = source_rest.copy()
        target_rest[3, 1] = -0.0005
        target_rest[[1, 2]] = [[-0.12, -0.08, 0], [0.12, -0.08, 0]]
        source_world = np.broadcast_to(np.eye(3), (1, 22, 3, 3)).copy()
        source_world[:, 0] = Rotation.from_euler("x", 10, degrees=True).as_matrix()
        legacy = solver.baseline(source_world, source_rest, target_rest)
        captured = source_world[:, 0]
        local = solver.baseline(source_world, source_rest, target_rest, pelvis_world=captured)

        root = np.array([[0.0, 1.0, 0.0]])
        _, legacy_world = solver.fk(legacy, target_rest, root)
        _, world = solver.fk(local, target_rest, root)
        np.testing.assert_allclose(world[:, 1:], legacy_world[:, 1:], atol=1e-12)
        np.testing.assert_allclose(world[:, 0], captured, atol=1e-12)

        pose = dict(world=world, rest=target_rest, root=root)
        fixed, _ = solver.characterize(pose, pose, None, source_world)
        np.testing.assert_allclose(fixed["root"], root, atol=1e-12)
        legacy_pose = dict(pose, world=legacy_world)
        lifted, _ = solver.characterize(legacy_pose, legacy_pose, None, source_world)
        self.assertGreater(lifted["root"][0, 1] - root[0, 1], 0.15)
