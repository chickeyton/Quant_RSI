"""Test-server sync: only missing store files move, nothing private leaves, TWS stays local."""

import io
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import pytest
from intraday_helpers import build_snapshot

from quant_rsi.core.artifacts import digest
from quant_rsi.intraday import sync
from quant_rsi.intraday.sync import RemoteError, ServerConfig, SshRemote

ROOT = Path(__file__).parents[1]


def server(**overrides) -> ServerConfig:
    value = {
        "schema_version": 1,
        "host": "test.example.org",
        "port": 2222,
        "user": "tester",
        "identity_file": "key",
        "workspace": "/workspace/tester",
    }
    return ServerConfig.model_validate({**value, **overrides})


class FakeRemote:
    """The server's workspace as a local directory; `python` runs this interpreter."""

    def __init__(self, root: Path, config: ServerConfig):
        self.root, self.config = root, config
        self.extracted, self.executed = [], []
        (self.local(config.repo)).mkdir(parents=True)

    def local(self, path: str) -> Path:
        return self.root / PurePosixPath(path).relative_to(self.config.workspace)

    def list_files(self, directory):
        base = self.local(directory)
        return {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}

    def extract(self, directory, archive, *, overwrite):
        base = self.local(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            names = tar.getnames()
            for member in tar.getmembers():
                if overwrite or not (base / member.name).exists():
                    tar.extract(member, base, filter="data")
        self.extracted.append((directory, sorted(names), overwrite))

    def read(self, path):
        return self.local(path).read_bytes()

    def execute(self, argv):
        self.executed.append(list(argv))
        command = [sys.executable, *argv[1:]] if argv[0] == "python" else list(argv)
        result = subprocess.run(command, cwd=self.local(self.config.repo), capture_output=True)
        return result.returncode, result.stdout


@pytest.fixture
def remote(tmp_path):
    return FakeRemote(tmp_path / "server", server())


def test_push_snapshot_sends_only_missing_files_and_verifies(tmp_path, remote):
    store, ref = build_snapshot(tmp_path, count=3)
    local = tmp_path / "store"
    (local / "pacing.sqlite").write_bytes(b"local pacing state")
    first = sync.push_snapshot(remote, remote.config, local, ref, "runs/ko/store")
    assert first["sent"] > 0 and first["already_present"] == 0
    assert first["verified_blocks"] == len(store.load(ref).blocks)
    sent = remote.extracted[0][1]
    assert all(name.startswith(("objects/", "requests/")) for name in sent)
    assert remote.extracted[0][2] is False  # stored objects are never overwritten
    again = sync.push_snapshot(remote, remote.config, local, ref, "runs/ko/store")
    assert again["sent"] == 0 and again["already_present"] == first["sent"]
    assert len(remote.extracted) == 1


def test_push_snapshot_refuses_a_corrupt_local_snapshot(tmp_path, remote):
    _, ref = build_snapshot(tmp_path, count=3)
    manifest = tmp_path / "store" / "objects" / ref[:2] / ref
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        sync.push_snapshot(remote, remote.config, tmp_path / "store", ref, "runs/ko/store")
    assert remote.extracted == []


@pytest.mark.parametrize("command", ["collect", "resolve", "ibkr-check", "paper-run", None])
def test_only_offline_commands_run_on_the_server(remote, command):
    with pytest.raises(ValueError, match="TWS machine"):
        sync.run_offline(remote, [command] if command else [])
    assert remote.executed == []


def test_cli_refuses_tws_commands_without_a_traceback(tmp_path, remote, capsys):
    config = tmp_path / "server.json"
    config.write_text(remote.config.model_dump_json())
    argv = ["--server", str(config), "run", "--", "collect", "--help"]
    assert sync.main(argv, remote=remote) == 2
    assert "TWS machine" in capsys.readouterr().err and remote.executed == []


def test_run_offline_wraps_the_intraday_cli(remote):
    code, _ = sync.run_offline(remote, ["evaluate", "--help"])
    assert code == 0 and remote.executed == [
        ["python", "-m", "quant_rsi.intraday", "evaluate", "--help"]
    ]


@pytest.mark.parametrize("path", ["../other", "/etc/passwd", "runs/../../x", "", "."])
def test_paths_stay_inside_the_workspace(remote, path):
    with pytest.raises(ValueError):
        remote.config.in_repo(path)
    with pytest.raises(ValueError):
        server(repo_dir=path or "..")


def test_push_code_refuses_private_files(tmp_path, remote):
    (tmp_path / "a.py").write_text("x = 1\n")
    result = sync.push_code(remote, remote.config, tmp_path, ["a.py"])
    assert result["files"] == 1 and remote.extracted[0][2] is True
    for private in (".quant_rsi/ibkr.json", ".env", "runs/ko/model.txt"):
        with pytest.raises(ValueError, match="private"):
            sync.push_code(remote, remote.config, tmp_path, ["a.py", private])


def test_pull_checks_digest_and_never_silently_overwrites(tmp_path, remote):
    model = remote.local(remote.config.in_repo("runs/ko/model.txt"))
    model.parent.mkdir(parents=True)
    model.write_bytes(b"tree\n")
    target = tmp_path / "local" / "model.txt"
    with pytest.raises(RemoteError, match="expected"):
        sync.pull(remote, remote.config, "runs/ko/model.txt", target, expect_sha256="0" * 64)
    assert not target.exists()
    result = sync.pull(
        remote, remote.config, "runs/ko/model.txt", target, expect_sha256=digest(b"tree\n")
    )
    assert result["sha256"] == digest(b"tree\n") and target.read_bytes() == b"tree\n"
    model.write_bytes(b"tree 2\n")
    with pytest.raises(FileExistsError):
        sync.pull(remote, remote.config, "runs/ko/model.txt", target)
    sync.pull(remote, remote.config, "runs/ko/model.txt", target, force=True)
    assert target.read_bytes() == b"tree 2\n"


def test_ssh_commands_quote_paths_and_activate_the_venv():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b"")

    remote = SshRemote(server(workspace="/workspace/a b"), runner=runner)
    remote.execute(["python", "-m", "pytest", "-k", "x; rm -rf /"])
    argv, kwargs = calls[0]
    assert argv[:5] == ["ssh", "-i", "key", "-p", "2222"] and argv[-2] == "tester@test.example.org"
    script = argv[-1]
    assert script.startswith("cd '/workspace/a b/repo/chickeyton/Quant_RSI' && . ")
    assert "'/workspace/a b/.venv-quant-rsi/bin/activate'" in script
    assert script.endswith("python -m pytest -k 'x; rm -rf /'")
    remote.extract("/workspace/a b/s", b"", overwrite=False)
    assert "--skip-old-files" in calls[-1][0][-1]
    failing = SshRemote(server(), runner=lambda a, **k: subprocess.CompletedProcess(a, 3, b""))
    with pytest.raises(RemoteError, match="exit status 3"):
        failing.read("/workspace/tester/x")


def test_checked_in_template_validates():
    ServerConfig.model_validate_json((ROOT / "examples/test_server.json.template").read_bytes())
