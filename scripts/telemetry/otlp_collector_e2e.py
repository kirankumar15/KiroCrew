#!/usr/bin/env python3
"""Opt-in end-to-end check: Kiro Crew -> a real OpenTelemetry Collector.

The two automated tiers in ``test/metrics/test_otlp_wire_e2e.py`` cover
serialization and the OTLP/HTTP wire without any external process, which is why
they can run in CI. What they cannot cover is anything that only exists once a
SEPARATE process is on the other end, and that gap is not hypothetical -- writing
this script is what surfaced two defects neither tier could reach: the release
asset you must fetch is not named what the obvious guess says (that URL 404s), and
the exporter's flush had to complete before another process could read the file it
wrote. Both live in the seam between two processes, which is exactly what tier 2,
decoding a payload this process produced, does not have.

A third motivation is real but so far unobserved: a payload can decode correctly
and still be rejected by a collector's own validation. Nothing in this repo has
hit that yet, so it is a reason to keep the tier rather than a reason it exists.

That check needs a downloaded binary and outbound network, so it cannot run in a
network-isolated CI and is deliberately NOT a test. Run it by hand:

    python3 scripts/telemetry/otlp_collector_e2e.py --expected-sha256 <digest>

It downloads a PINNED collector release, verifies it, runs it with a file
exporter, drives one real Kiro Crew export into it, and asserts every required
instrument plus the resource attributes arrived. Nothing is installed system-wide
and everything lands in a temp directory that is removed on exit.

SUPPLY CHAIN. Verifying against a digest you already trust is the DEFAULT, and the
script refuses to download anything without one. The weaker path is an explicit,
named opt-in:

    python3 scripts/telemetry/otlp_collector_e2e.py --trust-release-checksums

That accepts the ``checksums.txt`` published with the release, whose trust root is
only GitHub serving that release — the checksums come from the same origin as the
bytes they attest, so a compromised release can serve a matching pair. It exists
to bootstrap: it prints the observed digest, which you then pin. The refusal comes
BEFORE the download rather than after, because fetching first and only then
finding there is nothing to check against is how a "verified" script ends up
running an unverified binary. A digest mismatch is always fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

#: Pinned collector release. Bump deliberately, and re-pin any digest alongside.
COLLECTOR_VERSION = "0.109.0"
_RELEASE_BASE = (
    "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/"
    f"v{COLLECTOR_VERSION}"
)
_ARCHIVE_TEMPLATE = "otelcol_{version}_{os}_{arch}.tar.gz"
# The checksums asset is named after the RELEASE REPO and distribution, not after
# the archive — an "otelcol_<version>_checksums.txt" guess 404s. Each release also
# publishes a cosign .sig/.pem beside it; verifying those is the stronger trust
# root and the guide documents it.
_CHECKSUMS_NAME = "opentelemetry-collector-releases_otelcol_checksums.txt"

_ARCH_BY_MACHINE = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
}

_COLLECTOR_CONFIG = """\
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:{otlp_port}

exporters:
  file:
    path: {out_path}

service:
  telemetry:
    logs:
      level: warn
    metrics:
      level: none
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [file]
"""


def _log(message: str) -> None:
    print(f"[otlp-e2e] {message}", flush=True)


def _fail(message: str) -> "None":
    print(f"[otlp-e2e] FAIL: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def _archive_name() -> str:
    system = platform.system().lower()
    if system not in ("linux", "darwin"):
        _fail(f"unsupported platform {system!r}; run this on Linux or macOS")
    arch = _ARCH_BY_MACHINE.get(platform.machine().lower())
    if arch is None:
        _fail(f"unsupported architecture {platform.machine()!r}")
    return _ARCHIVE_TEMPLATE.format(version=COLLECTOR_VERSION, os=system, arch=arch)


def _download(url: str, dest: Path) -> None:
    """Fetch *url* to *dest*. Refuses any URL outside the pinned release.

    The prefix check makes "this script only ever fetches from one pinned
    collector release" an ENFORCED property rather than an incidental one: every
    caller builds its URL from :data:`_RELEASE_BASE`, and a future edit that
    introduced a second source would fail here instead of silently widening what
    this script will download and execute.
    """
    if not url.startswith(f"{_RELEASE_BASE}/"):
        _fail(f"refusing to fetch outside the pinned release: {url}")
    _log(f"downloading {url}")
    try:
        # The URL is not literal (the asset name is platform-dependent), but it is
        # constrained above to the pinned _RELEASE_BASE -- a hardcoded https GitHub
        # release tag -- and whatever comes back is SHA-256 verified before use.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        with urllib.request.urlopen(url, timeout=120) as response:
            dest.write_bytes(response.read())
    except Exception as exc:  # noqa: BLE001 -- a fetch failure is a plain abort
        _fail(f"download failed ({exc}); this script needs outbound network")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_from_checksums(text: str, archive_name: str) -> "str | None":
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == archive_name:
            return parts[0]
    return None


def _fetch_collector(workdir: Path, pinned_digest: "str | None", trust_release: bool) -> Path:
    archive_name = _archive_name()

    # Decide the trust root BEFORE downloading anything. Fetching first and only
    # then discovering there is nothing to check the bytes against is how a
    # "verified" script ends up running an unverified binary.
    if not pinned_digest and not trust_release:
        _fail(
            "no digest to verify against. Pass --expected-sha256 <digest> (the "
            "safe default), or --trust-release-checksums to accept the checksums "
            "file published alongside the archive. The latter is weaker: the "
            "checksums come from the same place as the bytes they attest, so a "
            "compromised release can serve a matching pair. Bootstrap with "
            "--trust-release-checksums once, then pin the digest it prints."
        )

    archive = workdir / archive_name
    _download(f"{_RELEASE_BASE}/{archive_name}", archive)

    observed = _sha256(archive)
    _log(f"sha256({archive_name}) = {observed}")

    if pinned_digest:
        if observed.lower() != pinned_digest.lower():
            _fail(f"digest mismatch: expected {pinned_digest}, got {observed}")
        _log("verified against the digest supplied on the command line")
    else:
        checksums = workdir / _CHECKSUMS_NAME
        _download(f"{_RELEASE_BASE}/{_CHECKSUMS_NAME}", checksums)
        expected = _expected_from_checksums(checksums.read_text(encoding="utf-8"), archive_name)
        if expected is None:
            _fail(f"{archive_name} is not listed in {_CHECKSUMS_NAME}")
        if observed.lower() != expected.lower():
            _fail(f"digest mismatch: checksums.txt says {expected}, got {observed}")
        _log(
            "verified against the release checksums file (WEAK trust root: same "
            f"origin as the archive). Pin this for later runs: --expected-sha256 {observed}"
        )

    with tarfile.open(archive) as tar:
        member = next((m for m in tar.getmembers() if Path(m.name).name == "otelcol"), None)
        if member is None:
            _fail("archive does not contain an 'otelcol' binary")
        # Refuse anything that is not a regular file BEFORE extracting. A symlink
        # or hardlink member named 'otelcol' can point outside the destination,
        # and on the 3.10 leg below there is no stdlib filter to catch it.
        if not member.isreg():
            _fail(f"archive member 'otelcol' is not a regular file: {member.type!r}")
        # Extract the single known member under a LITERAL name rather than
        # extractall(), so no member name can traverse out of the temp dir.
        member.name = "otelcol"
        try:
            tar.extract(member, path=workdir, filter="data")
        except TypeError:
            # Python 3.10 has no `filter=` keyword (added in 3.12, backported to
            # 3.10.12). Same TypeError-fallback shape the repo already uses in
            # papyrus/backend/tectonic.py::_extract_tar. Safe here because the two
            # things `filter="data"` would have protected against are already
            # handled above: the name is a literal, and non-regular members are
            # refused.
            tar.extract(member, path=workdir)
    binary = workdir / "otelcol"
    binary.chmod(0o755)
    return binary


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, process: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _fail(f"collector exited early with code {process.returncode}")
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    _fail(f"collector did not accept connections on 127.0.0.1:{port} within {timeout}s")


def _export_once(endpoint: str, metrics_dir: Path) -> None:
    """Drive one real Kiro Crew export at *endpoint* via the production seam."""
    from unittest.mock import patch

    os.environ["KIROCREW_TELEMETRY"] = "1"
    from kiro_crew.config.loader import KiroCrewConfig
    from kiro_crew.metrics import inventory_gauges as ig
    from kiro_crew.metrics import provider as pm

    # Install-scoped inventory publishes only from the elected reporter (normally
    # the gateway). This script IS the only process exporting here, so it claims
    # the role; without it every inventory instrument would correctly report as
    # absent and the run would look like a collector failure.
    ig.mark_install_reporter()

    cfg = KiroCrewConfig()
    cfg.telemetry.enabled = True
    cfg.telemetry.otlp_endpoint = endpoint
    cfg.telemetry.export_interval_seconds = 3600  # force_flush is the only export

    pm.reset_for_testing()
    try:
        with patch.object(KiroCrewConfig, "load", return_value=cfg):
            with patch.object(pm, "_default_metrics_dir", return_value=metrics_dir):
                recorder = pm.get_recorder()
                if not recorder.enabled:
                    _fail("telemetry did not enable; is the [otlp] extra installed?")
                provider = pm._provider
                if provider is None:
                    _fail("live build produced no MeterProvider")
                provider.force_flush()
    finally:
        pm.reset_for_testing()


def _collected(out_path: Path) -> tuple[set[str], dict[str, str]]:
    """Metric names and resource attributes the collector wrote out."""
    if not out_path.exists():
        _fail(f"collector wrote no output at {out_path}")
    names: set[str] = set()
    resource: dict[str, str] = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            # A truncated trailing line can survive even a graceful shutdown if
            # the process was killed. Skip it rather than abort: the assertion
            # that matters is whether the roster arrived, and a partial last line
            # cannot remove a name an earlier complete line already carried.
            _log("skipping one unparseable output line (truncated flush)")
            continue
        for rm in payload.get("resourceMetrics") or []:
            for attr in (rm.get("resource") or {}).get("attributes") or []:
                value = attr.get("value") or {}
                resource[attr.get("key", "")] = str(
                    value.get("stringValue", next(iter(value.values()), ""))
                )
            for sm in rm.get("scopeMetrics") or []:
                for metric in sm.get("metrics") or []:
                    if metric.get("name"):
                        names.add(metric["name"])
    return names, resource


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-sha256",
        default=os.environ.get("OTELCOL_EXPECTED_SHA256"),
        help="pin the collector archive digest instead of trusting checksums.txt",
    )
    parser.add_argument(
        "--trust-release-checksums",
        action="store_true",
        help=(
            "accept the checksums file published with the archive instead of a "
            "pinned digest (WEAKER: same origin as the bytes it attests). Use once "
            "to bootstrap, then pin the digest it prints."
        ),
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the temp workdir (collector binary, config, captured output)",
    )
    args = parser.parse_args()

    from kiro_crew.metrics import inventory_gauges as ig
    from kiro_crew.metrics import process_gauges as pg

    workdir = Path(tempfile.mkdtemp(prefix="kirocrew-otlp-e2e-"))
    _log(f"workdir {workdir}")
    process = None
    try:
        binary = _fetch_collector(workdir, args.expected_sha256, args.trust_release_checksums)

        otlp_port = _free_port()
        out_path = workdir / "collected.json"
        config_path = workdir / "collector.yaml"
        config_path.write_text(
            _COLLECTOR_CONFIG.format(otlp_port=otlp_port, out_path=out_path),
            encoding="utf-8",
        )

        _log(f"starting collector on 127.0.0.1:{otlp_port}")
        # The repo's own kirocrew.download-then-subprocess rule states the
        # required mitigation for executing a downloaded file: verified integrity
        # before execution. _fetch_collector does exactly that and aborts on any
        # mismatch, so reaching this line means the bytes matched a checksum from
        # the pinned release. Re-assert the two structural facts here so the exec
        # site itself carries them rather than trusting a caller far above.
        if binary.parent != workdir or not binary.is_file():
            _fail(f"refusing to execute {binary}: not the verified binary in {workdir}")
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
        # argv is a list (never a shell string); every element is either a literal
        # or a path under the temp workdir this process created and verified above.
        process = subprocess.Popen(
            # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
            [str(binary), "--config", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Pin the decode instead of inheriting the platform's locale codec, so
            # collector output cannot come back mojibake (or raise) on a host whose
            # default is not UTF-8. errors="replace" keeps a stray byte from
            # turning a diagnostic read into an exception.
            encoding="utf-8",
            errors="replace",
        )
        _wait_for_port(otlp_port, process)

        endpoint = f"http://127.0.0.1:{otlp_port}/v1/metrics"
        _log(f"exporting into {endpoint}")
        _export_once(endpoint, workdir / "local-metrics")

        # Shut the collector down BEFORE reading. The file exporter buffers and
        # flushes asynchronously, so reading while it is still running catches a
        # half-written line; a graceful terminate makes it flush and close first.
        _log("stopping collector to flush its file exporter")
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

        names, resource = _collected(out_path)
        _log(f"collector received {len(names)} instruments")

        expected = set(pg.ALL_METRIC_NAMES) | set(ig.ALL_METRIC_NAMES)
        # A probe that cannot answer yields NO observation by design, so the
        # required set must exclude every instrument whose source is legitimately
        # absent on a healthy host. Otherwise this script reports a collector
        # failure for running on macOS (no /proc thread or fd surface), or on an
        # install that has never ingested a knowledge document, configured an MCP
        # server, or armed a monitor loop -- none of which says anything about
        # whether the collector accepted what we sent.
        optional = {
            pg.GAUGE_THREADS_OS,  # Linux-only surface
            pg.GAUGE_OPEN_FDS,  # Linux-only surface
            ig.GAUGE_MONITOR_LOOPS_ACTIVE,  # no auto-nudge service in this process
            ig.GAUGE_KNOWLEDGE_DOCUMENTS,  # absent until something is ingested
            ig.GAUGE_MCP_SERVERS,  # absent when no roster is configured
            ig.COUNTER_PROBE_FAILURES,  # absent by design while every probe answers
        }
        missing = sorted((expected - optional) - names)
        if missing:
            _fail(f"instruments never reached the collector: {missing}")
        for name in sorted(optional - names):
            _log(f"  -- {name} (no observation; source absent on this host)")
        if resource.get("service.name") != "kirocrew":
            _fail(f"resource attributes wrong on the wire: {resource}")

        _log("resource attributes: " + ", ".join(sorted(resource)))
        for name in sorted(names):
            _log(f"  ok {name}")
        _log("PASS -- every required instrument reached a real collector")
        return 0
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if args.keep:
            _log(f"kept workdir {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
