"""Exercise cache relocation with real CMake; only GPU preflight commands are stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class PostCreateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tidal cache test ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "host" / "src" / "tidal.cpp"
        (self.project / ".devcontainer").mkdir(parents=True)
        shutil.copy2(REPO / ".devcontainer/post-create.sh", self.project / ".devcontainer")
        presets = json.loads((REPO / "CMakePresets.json").read_text())
        # Cache relocation is generator-independent; no Ninja/CUDA SDK is needed here.
        presets["configurePresets"][0]["generator"] = "Unix Makefiles"
        (self.project / "CMakePresets.json").write_text(json.dumps(presets))
        (self.project / "dependency").mkdir()
        (self.project / "dependency/CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\nproject(dependency NONE)\n")
        (self.project / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\nproject(cache_probe NONE)\n"
            "execute_process(COMMAND \"${CMAKE_COMMAND}\" "
            "-S \"${CMAKE_CURRENT_SOURCE_DIR}/dependency\" "
            "-B \"${CMAKE_CURRENT_BINARY_DIR}/_deps/probe-subbuild\" "
            "COMMAND_ERROR_IS_FATAL ANY)\n")
        commands = self.root / "commands"
        commands.mkdir()
        for name, script in {
            "nvcc": "#!/bin/sh\nexit 0\n",
            "nvidia-smi": '#!/bin/sh\nexit "${TIDAL_TEST_GPU_STATUS:-0}"\n',
        }.items():
            executable = commands / name
            executable.write_text(script)
            executable.chmod(0o755)
        self.env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"],
                        TIDAL_TEST_GPU_STATUS="0")

    @property
    def build(self):
        return self.project / "build/dev-cuda"

    def command(self, *args, success=True):
        result = subprocess.run(args, cwd=self.project, env=self.env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def setup_container(self, **kwargs):
        return self.command("bash", ".devcontainer/post-create.sh", **kwargs)

    def backups(self):
        return list((self.project / "build").glob("dev-cuda.stale.*"))

    def relocate_checkout(self):
        destination = self.root / "workspaces" / "tidal.cpp"
        destination.parent.mkdir()
        self.project.rename(destination)
        self.project = destination

    def test_fresh_and_repeated_setup_reuse_the_build(self):
        self.setup_container()
        marker = self.build / "local-artifact.bin"
        marker.write_bytes(b"keep incremental build")
        self.setup_container()
        self.assertEqual(marker.read_bytes(), b"keep incremental build")
        self.assertEqual(self.backups(), [])

    def test_relocated_checkout_preserves_build_and_refreshes_nested_cache(self):
        self.command("cmake", "--preset", "dev-cuda")
        caches = [Path("CMakeCache.txt"), Path("_deps/probe-subbuild/CMakeCache.txt")]
        original = {path: (self.build / path).read_bytes() for path in caches}
        (self.build / "local-artifact.bin").write_bytes(b"preserve old build")
        self.relocate_checkout()
        self.setup_container()
        backups = self.backups()
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "local-artifact.bin").read_bytes(), b"preserve old build")
        for path in caches:
            self.assertEqual((backups[0] / path).read_bytes(), original[path])
            self.assertIn(str(self.project), (self.build / path).read_text())
            self.assertNotIn(str(self.root / "host"), (self.build / path).read_text())
        self.setup_container()
        self.assertEqual(self.backups(), backups)

    def test_relocated_build_with_unchanged_source_is_refreshed(self):
        previous_build = self.project / "build/host-build"
        self.command("cmake", "--preset", "dev-cuda", "-B", str(previous_build))
        old_cache = (previous_build / "CMakeCache.txt").read_bytes()
        previous_build.rename(self.build)
        self.setup_container()
        backups = self.backups()
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "CMakeCache.txt").read_bytes(), old_cache)

    def test_gpu_preflight_failure_does_not_touch_stale_build(self):
        self.command("cmake", "--preset", "dev-cuda")
        old_cache = (self.build / "CMakeCache.txt").read_bytes()
        self.relocate_checkout()
        self.env["TIDAL_TEST_GPU_STATUS"] = "1"
        result = self.setup_container(success=False)
        self.assertIn("GPU access is unavailable", result.stdout)
        self.assertEqual((self.build / "CMakeCache.txt").read_bytes(), old_cache)
        self.assertEqual(self.backups(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
