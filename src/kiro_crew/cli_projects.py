"""CLI handlers for portable Project bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kiro_crew.project_capabilities import ProjectCapabilityError, ProjectCapabilityManager
from kiro_crew.project_git import GitProjectStore, ProjectGitError
from kiro_crew.project_manifest import ProjectManifestError, create_project_manifest
from kiro_crew.project_registry import ProjectRegistry, ProjectRegistryError
from kiro_crew.sel import sel


def _print_project(project_id: str, name: str, origin: str, path: str) -> None:
    print(f"{project_id}  {name}  {origin}  {path}")


def _handle_project(args: argparse.Namespace) -> None:
    """Dispatch ``kirocrew project`` bundle-management commands."""
    registry = ProjectRegistry()
    capability_manager = ProjectCapabilityManager(registry)
    try:
        if args.project_action == "create":
            manifest = create_project_manifest(args.path, name=args.name)
            project = registry.add_local(args.path)
            print(f"Created Project {project.name} ({manifest.id}) at {args.path}")
        elif args.project_action == "add":
            local_source = Path(args.source).expanduser()
            if local_source.exists():
                project = capability_manager.register_local(local_source)
            else:
                project = GitProjectStore(registry).add(
                    args.source,
                    before_primary_change=capability_manager.guard_primary_change,
                )
            print(f"Added Project {project.name} ({project.id}) from {args.source}")
        elif args.project_action == "list":
            projects = registry.list_projects()
            if not projects:
                print("No Projects registered.")
                return
            print("ID  NAME  ORIGIN  PATH")
            for project in projects:
                primary = project.registrations[-1]
                _print_project(
                    project.id,
                    project.name,
                    primary.origin,
                    str(primary.path),
                )
        elif args.project_action == "show":
            project = registry.resolve(args.identifier)
            print(f"Name: {project.name}")
            print(f"ID: {project.id}")
            print("Materializations:")
            for registration in project.registrations:
                print(f"  {registration.origin}: {registration.path}")
        elif args.project_action == "sync":
            project = GitProjectStore(registry).sync(args.identifier)
            capability_status = ProjectCapabilityManager(registry).refresh_if_active(project.id)
            if capability_status.active:
                from kiro_crew.agent import rebuild_agent_config

                rebuild_agent_config()
            sel().log_api_access(
                caller="local-cli",
                operation="project_sync",
                outcome="allowed",
                source="cli",
                resources=f"project={project.id}",
            )
            print(f"Synced Project {project.name} ({project.id})")
        else:
            raise ProjectRegistryError("choose a project command: create, add, list, show, or sync")
    except (
        OSError,
        ProjectCapabilityError,
        ProjectGitError,
        ProjectManifestError,
        ProjectRegistryError,
    ) as exc:
        print(f"Project error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
