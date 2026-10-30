"""Move offline work to a shared Linux test server: `python -m quant_rsi.intraday.sync`.

Only the machine running TWS talks to IBKR (`resolve`, `collect`, `paper-*`). The server gets
copies of the code and snapshots and runs offline work only: tests, `evaluate` and `fit`.
The TWS API port is therefore never reachable from a machine other people share.

Snapshots are content-addressed, so a push sends only missing files and never overwrites one;
the server then loads the snapshot, which verifies every digest. Pulled files can be checked
against the SHA-256 that `fit` printed.
"""

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from pydantic import Field, field_validator

from quant_rsi.core.artifacts import digest
from quant_rsi.core.contracts_base import StrictModel

# Commands that read only stored files. Everything else connects to TWS.
OFFLINE_COMMANDS = frozenset({"evaluate", "fit"})
# Never sent to a shared server, even if Git would include them.
PRIVATE_PREFIXES = (".quant_rsi/", ".env", "secrets/", "credentials/", "runs/", "data/")
STORE_DIRECTORIES = ("objects", "requests")

VERIFY_SNAPSHOT = """
import sys
from pathlib import Path
from quant_rsi.intraday.storage import SnapshotStore
snapshot = SnapshotStore(Path(sys.argv[1])).load(sys.argv[2])
print(len(snapshot.blocks))
"""


class RemoteError(RuntimeError):
    pass


def relative(value: str) -> str:
    """A non-empty POSIX path that stays inside its base directory."""
    path = PurePosixPath(value.replace("\\", "/"))
    if not value or path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError(f"path must be relative and stay inside the workspace: {value!r}")
    return str(path)


class ServerConfig(StrictModel):
    """Test server settings; stored outside version control in `.quant_rsi/test_server.json`."""

    schema_version: Literal[1]
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, gt=0, lt=65536)
    user: str = Field(min_length=1, max_length=64)
    identity_file: str = Field(min_length=1)
    workspace: str
    repo_dir: str = "repo/chickeyton/Quant_RSI"
    venv: str = ".venv-quant-rsi"
    profile: str = "profile"
    # Temporary files of remote commands (pytest's tmp_path included) stay in the workspace.
    tmp_dir: str = "tmp"

    @field_validator("workspace")
    @classmethod
    def absolute(cls, value):
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("workspace must be an absolute POSIX path")
        return str(path)

    @field_validator("repo_dir", "venv", "profile", "tmp_dir")
    @classmethod
    def inside(cls, value):
        return relative(value)

    def path(self, *parts: str) -> str:
        """Absolute server path under the workspace."""
        return str(PurePosixPath(self.workspace).joinpath(*parts))

    @property
    def repo(self) -> str:
        return self.path(self.repo_dir)

    def in_repo(self, value: str) -> str:
        return str(PurePosixPath(self.repo) / relative(value))


class Remote(Protocol):
    def list_files(self, directory: str) -> set[str]: ...

    def extract(self, directory: str, archive: bytes, *, overwrite: bool) -> None: ...

    def read(self, path: str) -> bytes: ...

    def execute(self, argv: Sequence[str]) -> tuple[int, bytes]: ...


class SshRemote:
    """Runs each operation as one `ssh` command; the server needs a POSIX shell and GNU tar."""

    def __init__(self, server: ServerConfig, *, ssh: str = "ssh", runner=subprocess.run):
        self.server, self.ssh, self.runner = server, ssh, runner

    def argv(self, script: str) -> list[str]:
        s = self.server
        return [
            self.ssh,
            "-i",
            s.identity_file,
            "-p",
            str(s.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            f"{s.user}@{s.host}",
            script,
        ]

    def _run(self, script: str, stdin: bytes | None = None) -> tuple[int, bytes]:
        # stderr is passed through so remote progress and errors stay visible.
        result = self.runner(self.argv(script), input=stdin, stdout=subprocess.PIPE, check=False)
        return result.returncode, result.stdout

    def _checked(self, script: str, stdin: bytes | None = None) -> bytes:
        code, out = self._run(script, stdin)
        if code:
            raise RemoteError(f"remote command failed with exit status {code}")
        return out

    def list_files(self, directory: str) -> set[str]:
        d = shlex.quote(directory)
        out = self._checked(f"if [ -d {d} ]; then cd {d} && find . -type f -print0; fi")
        return {name.decode()[2:] for name in out.split(b"\0") if name}

    def extract(self, directory: str, archive: bytes, *, overwrite: bool) -> None:
        d = shlex.quote(directory)
        keep = "" if overwrite else " --skip-old-files"
        self._checked(f"mkdir -p {d} && tar -x{keep} -f - -C {d}", archive)

    def read(self, path: str) -> bytes:
        return self._checked(f"cat {shlex.quote(path)}")

    def execute(self, argv: Sequence[str]) -> tuple[int, bytes]:
        s = self.server
        command = " ".join(shlex.quote(a) for a in argv)
        tmp = shlex.quote(s.path(s.tmp_dir))
        script = (
            f"mkdir -p {tmp} && export TMPDIR={tmp} && "
            f"cd {shlex.quote(s.repo)} && . {shlex.quote(s.path(s.profile))} && "
            f". {shlex.quote(s.path(s.venv, 'bin', 'activate'))} && {command}"
        )
        return self._run(script)


def archive(root: Path, names: Iterable[str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(names):
            tar.add(root / name, arcname=name, recursive=False)
    return buffer.getvalue()


def push_code(remote: Remote, server: ServerConfig, names: Sequence[str], code: bytes) -> dict:
    """Send a `git_snapshot` archive; credentials and local output are refused."""
    names = [relative(name) for name in names]
    private = [n for n in names if n.startswith(PRIVATE_PREFIXES)]
    if private:
        raise ValueError(f"refusing to send private files: {private[:5]}")
    remote.extract(server.repo, code, overwrite=True)
    return {"repo": server.repo, "files": len(names)}


def git_snapshot(root: Path) -> tuple[list[str], bytes]:
    """Tracked and unignored working-tree files, with the bytes Git would commit.

    A Windows checkout may hold CRLF line endings; tests that fingerprint repository files
    expect the normalised bytes, so the archive comes from a temporary index, not the disk.
    """
    with tempfile.TemporaryDirectory() as temporary:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}

        def git(*args: str) -> bytes:
            return subprocess.run(
                ["git", *args], cwd=root, env=env, stdout=subprocess.PIPE, check=True
            ).stdout

        git("read-tree", "HEAD")
        git("-c", "core.safecrlf=false", "add", "--all")
        tree = git("write-tree").decode().strip()
        listing = git("ls-tree", "-r", "-z", "--name-only", tree)
        names = [name.decode() for name in listing.split(b"\0") if name]
        # `git archive` applies checkout line-ending conversion unless told otherwise.
        lf = ("-c", "core.autocrlf=false", "-c", "core.eol=lf")
        return names, git(*lf, "archive", "--format=tar", tree)


def push_snapshot(
    remote: Remote, server: ServerConfig, store: Path, snapshot_ref: str, remote_store: str
) -> dict:
    """Send the store's missing objects and request receipts, then verify on the server."""
    from quant_rsi.intraday.storage import SnapshotStore

    SnapshotStore(store).load(snapshot_ref)  # never ship a snapshot that fails locally
    local = {
        path.relative_to(store).as_posix()
        for directory in STORE_DIRECTORIES
        for path in (store / directory).rglob("*")
        if path.is_file()
    }
    target = server.in_repo(remote_store)
    missing = local - remote.list_files(target)
    if missing:
        remote.extract(target, archive(store, missing), overwrite=False)
    code, out = remote.execute(
        ["python", "-c", VERIFY_SNAPSHOT, relative(remote_store), snapshot_ref]
    )
    if code:
        raise RemoteError(f"snapshot {snapshot_ref} failed verification on the server")
    return {
        "remote_store": target,
        "sent": len(missing),
        "already_present": len(local) - len(missing),
        "verified_blocks": int(out.decode().strip()),
    }


def run_offline(remote: Remote, args: Sequence[str]) -> tuple[int, bytes]:
    if not args or args[0] not in OFFLINE_COMMANDS:
        raise ValueError(
            f"only {sorted(OFFLINE_COMMANDS)} run on the server; commands that connect to "
            "TWS run on the TWS machine"
        )
    return remote.execute(["python", "-m", "quant_rsi.intraday", *args])


def pull(
    remote: Remote,
    server: ServerConfig,
    source: str,
    destination: Path,
    *,
    expect_sha256: str | None = None,
    force: bool = False,
) -> dict:
    data = remote.read(server.in_repo(source))
    found = digest(data)
    if expect_sha256 and found != expect_sha256:
        raise RemoteError(f"{source} has SHA-256 {found}, expected {expect_sha256}")
    if destination.exists() and destination.read_bytes() != data and not force:
        raise FileExistsError(f"{destination} exists with different content; pass --force")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {"path": str(destination), "sha256": found, "bytes": len(data)}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m quant_rsi.intraday.sync")
    root.add_argument("--server", default=".quant_rsi/test_server.json")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("push-code")
    snapshot = sub.add_parser("push-snapshot")
    snapshot.add_argument("--store", required=True)
    snapshot.add_argument("--snapshot", required=True)
    snapshot.add_argument("--remote-store", help="repo-relative; defaults to --store")
    run = sub.add_parser("run", help="evaluate or fit on the server")
    run.add_argument("--save", help="also write the command's output to this local file")
    run.add_argument("args", nargs=argparse.REMAINDER)
    test = sub.add_parser("test", help="pytest on the server")
    test.add_argument("args", nargs=argparse.REMAINDER)
    get = sub.add_parser("pull")
    get.add_argument("--remote", required=True, help="repo-relative server path")
    get.add_argument("--local", required=True)
    get.add_argument("--expect-sha256")
    get.add_argument("--force", action="store_true")
    return root


def _rest(args: Sequence[str]) -> list[str]:
    return list(args[1:] if args and args[0] == "--" else args)


def main(argv=None, *, remote: Remote | None = None) -> int:
    try:
        return _main(argv, remote)
    except (ValueError, RemoteError, FileExistsError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2


def _main(argv, remote: Remote | None) -> int:
    args = parser().parse_args(argv)
    server = ServerConfig.model_validate_json(Path(args.server).read_bytes())
    remote = remote or SshRemote(server)
    root = Path.cwd()
    if args.command == "push-code":
        result = push_code(remote, server, *git_snapshot(root))
    elif args.command == "push-snapshot":
        result = push_snapshot(
            remote,
            server,
            Path(args.store),
            args.snapshot,
            args.remote_store or Path(args.store).as_posix(),
        )
    elif args.command in {"run", "test"}:
        rest = _rest(args.args)
        if args.command == "run":
            code, out = run_offline(remote, rest)
        else:
            code, out = remote.execute(["python", "-m", "pytest", *rest])
        if args.command == "run" and args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save).write_bytes(out)
        sys.stdout.write(out.decode(errors="replace"))
        return code
    else:
        result = pull(
            remote,
            server,
            args.remote,
            Path(args.local),
            expect_sha256=args.expect_sha256,
            force=args.force,
        )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
