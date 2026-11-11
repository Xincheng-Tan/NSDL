"""MAT-THOR adapter utilities.

The adapter is pure Python and intentionally does not import ai2thor.
"""

from nsdl.adapters.mat_thor.dataset_index import MatThorTask, iter_tasks
from nsdl.adapters.mat_thor.scene_encoder import encode_task

__all__ = [
    "MatThorTask",
    "encode_task",
    "iter_tasks",
]

