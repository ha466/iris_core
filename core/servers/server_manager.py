"""
server_manager.py - Manages prebuilt llama-server and parakeet-server subprocesses.
"""
from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import time
from typing import Optional

import httpx

import config

log = logging.getLogger("server_manager")


class ServerManager:
    """Orchestrates local llama-server and parakeet-server child processes."""

    def __init__(self):
        self.llama_proc: Optional[subprocess.Popen] = None
        self.parakeet_proc: Optional[subprocess.Popen] = None
        self._cleaned_up = False
        atexit.register(self.stop_all)

    @staticmethod
    def is_port_open(host: str, port: int) -> bool:
        """Check if a port is currently listening."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0

    def start_llama_server(self) -> None:
        """Start llama-server.exe if not already running."""
        if self.is_port_open(config.LLAMA_HOST, config.LLAMA_PORT):
            log.info(f"llama-server already running on {config.LLAMA_HOST}:{config.LLAMA_PORT}")
            return

        if not config.LLAMA_SERVER_EXE.exists():
            raise FileNotFoundError(f"llama-server binary not found: {config.LLAMA_SERVER_EXE}")
        if not config.LLAMA_MODEL_PATH.exists():
            raise FileNotFoundError(f"LLM model not found: {config.LLAMA_MODEL_PATH}")

        cmd = [
            str(config.LLAMA_SERVER_EXE),
            "-m", str(config.LLAMA_MODEL_PATH),
            "--host", config.LLAMA_HOST,
            "--port", str(config.LLAMA_PORT),
            "-c", str(config.LLAMA_CTX_SIZE),
            "--threads", str(config.LLAMA_THREADS),
            "--jinja",
            "--cache-type-k", config.LLAMA_KV_CACHE_TYPE,
            "--cache-type-v", config.LLAMA_KV_CACHE_TYPE,
        ]
        if not getattr(config, "LLAMA_THINKING_ENABLED", False):
            cmd.extend(["--reasoning-budget", "0"])
        log.info(f"Starting llama-server: {' '.join(cmd)}")
        self.llama_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

    def start_parakeet_server(self) -> None:
        """Start parakeet-server.exe if not already running."""
        if self.is_port_open(config.PARAKEET_HOST, config.PARAKEET_PORT):
            log.info(f"parakeet-server already running on {config.PARAKEET_HOST}:{config.PARAKEET_PORT}")
            return

        if not config.PARAKEET_SERVER_EXE.exists():
            raise FileNotFoundError(f"parakeet-server binary not found: {config.PARAKEET_SERVER_EXE}")
        if not config.PARAKEET_MODEL_PATH.exists():
            raise FileNotFoundError(f"Parakeet model not found: {config.PARAKEET_MODEL_PATH}")

        cmd = [
            str(config.PARAKEET_SERVER_EXE),
            "--model", str(config.PARAKEET_MODEL_PATH),
            "--host", config.PARAKEET_HOST,
            "--port", str(config.PARAKEET_PORT),
            "--threads", str(config.PARAKEET_THREADS),
        ]
        log.info(f"Starting parakeet-server: {' '.join(cmd)}")
        self.parakeet_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

    def wait_for_ready(self, timeout_secs: float = 30.0) -> bool:
        """Poll servers until both are responding to requests."""
        deadline = time.time() + timeout_secs
        llama_ready = False
        parakeet_ready = False

        while time.time() < deadline:
            # Check llama-server
            if not llama_ready:
                try:
                    r = httpx.get(f"http://{config.LLAMA_HOST}:{config.LLAMA_PORT}/health", timeout=1.0)
                    if r.status_code == 200:
                        llama_ready = True
                        log.info("llama-server is healthy and ready.")
                except Exception:
                    pass

            # Check parakeet-server
            if not parakeet_ready:
                if self.is_port_open(config.PARAKEET_HOST, config.PARAKEET_PORT):
                    parakeet_ready = True
                    log.info("parakeet-server is healthy and ready.")

            if llama_ready and parakeet_ready:
                return True

            time.sleep(0.5)

        raise TimeoutError(f"Servers failed to ready within {timeout_secs}s. Llama: {llama_ready}, Parakeet: {parakeet_ready}")

    def start_all(self, timeout_secs: float = 30.0) -> None:
        """Start both inference servers and await readiness."""
        self.start_llama_server()
        self.start_parakeet_server()
        self.wait_for_ready(timeout_secs=timeout_secs)

    def stop_all(self) -> None:
        """Gracefully terminate both subprocesses."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        for name, proc in [("parakeet-server", self.parakeet_proc), ("llama-server", self.llama_proc)]:
            if proc and proc.poll() is None:
                log.info(f"Stopping {name} (PID {proc.pid})...")
                try:
                    proc.terminate()
                    proc.wait(timeout=3.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        log.info("All server subprocesses stopped.")
