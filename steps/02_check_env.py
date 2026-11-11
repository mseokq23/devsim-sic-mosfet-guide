"""02. Environment check: DEVSIM import, math library, versions, thread settings, one tiny solve.

  python steps/02_check_env.py
"""
import contextlib
import io
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy, scipy, matplotlib
import devsim as ds

info = ds.get_parameter(name="info")
print("python    ", platform.python_version(), platform.machine())
print("devsim    ", info["version"])
print("solver    ", info["direct_solver"], "| math libraries:", info["math_libraries"])
print("extended  ", info["extended_precision"], "(128-bit floating point available)")
print("numpy     ", numpy.__version__, "| scipy", scipy.__version__, "| matplotlib", matplotlib.__version__)
print("threads   ", "MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"),
      "| OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"), "| cpu count", os.cpu_count())

# smallest possible DEVSIM problem: Laplace equation on a 1D line, V = 1 V at the left end
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):          # mesh messages are not needed here
    ds.create_1d_mesh(mesh="m")
    ds.add_1d_mesh_line(mesh="m", pos=0.0, ps=0.1, tag="left")
    ds.add_1d_mesh_line(mesh="m", pos=1.0, ps=0.1, tag="right")
    ds.add_1d_region(mesh="m", material="x", region="r", tag1="left", tag2="right")
    ds.add_1d_contact(mesh="m", name="left", tag="left", material="metal")
    ds.add_1d_contact(mesh="m", name="right", tag="right", material="metal")
    ds.finalize_mesh(mesh="m")
    ds.create_device(mesh="m", device="d")
ds.node_solution(device="d", region="r", name="V")
ds.edge_from_node_model(device="d", region="r", node_model="V")
ds.edge_model(device="d", region="r", name="flux", equation="(V@n0 - V@n1)*EdgeInverseLength")
ds.edge_model(device="d", region="r", name="flux:V@n0", equation="EdgeInverseLength")
ds.edge_model(device="d", region="r", name="flux:V@n1", equation="-EdgeInverseLength")
ds.equation(device="d", region="r", name="Laplace", variable_name="V", edge_model="flux")
for c, v in (("left", 1.0), ("right", 0.0)):
    ds.contact_node_model(device="d", contact=c, name=f"bc_{c}", equation=f"V - {v}")
    ds.contact_node_model(device="d", contact=c, name=f"bc_{c}:V", equation="1")
    ds.contact_equation(device="d", contact=c, name="Laplace", node_model=f"bc_{c}")
ds.solve(type="dc", absolute_error=1e-12, relative_error=1e-12, maximum_iterations=5)
v = ds.get_node_model_values(device="d", region="r", name="V")
print(f"1D Laplace: V = {[round(x, 3) for x in v]}  ({time.time() - t0:.2f} s)")
print("OK")
