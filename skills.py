from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from memory import get_data_root


class Skills:
    FILENAME = "SKILL.md"

    _cache: Dict[str, str] = {}
    _paths: Dict[str, Path] = {}
    _scanned_cwd: Optional[Path] = None

    @classmethod
    def search_dirs(cls) -> List[Path]:
        """
        Skill locations, highest precedence first.

        Skills come from the project the user is working in (cwd), so an
        installed CLI picks up that project's skills. Global skills in the
        data root (~/.pi-python/skills, or the repo's .pi-python in dev)
        are available everywhere.
        """
        workspace = Path.cwd()
        candidates = [
            workspace / ".pi-python" / "skills",
            workspace / "skills",
            get_data_root() / "skills",
        ]

        seen: set[Path] = set()
        dirs: List[Path] = []
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                dirs.append(path)
        return dirs

    @classmethod
    def refresh(cls) -> None:
        """Reload all skills into memory, supporting skills/<skill_name>/SKILL.md structure."""
        cls._cache.clear()
        cls._paths.clear()
        cls._scanned_cwd = Path.cwd()

        for base_dir in cls.search_dirs():
            if not base_dir.is_dir():
                continue

            # 1. Folder structure: skills/<skill_name>/SKILL.md
            for folder in sorted(base_dir.iterdir()):
                if not folder.is_dir() or folder.name.startswith("."):
                    continue

                skill_name = folder.name
                if skill_name in cls._cache:
                    continue  # earlier dir wins

                skill_file = folder / cls.FILENAME
                if not skill_file.exists():
                    skill_file = folder / "skill.md"
                if not skill_file.exists():
                    md_files = sorted(folder.glob("*.md"))
                    if md_files:
                        skill_file = md_files[0]

                if skill_file.is_file():
                    try:
                        cls._cache[skill_name] = skill_file.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    cls._paths[skill_name] = skill_file

            # 2. Direct file structure: skills/<skill_name>.md
            for file in sorted(base_dir.glob("*.md")):
                if not file.is_file() or file.stem.upper() == "SKILL":
                    continue
                skill_name = file.stem
                if skill_name in cls._cache:
                    continue
                try:
                    cls._cache[skill_name] = file.read_text(encoding="utf-8")
                except OSError:
                    continue
                cls._paths[skill_name] = file

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Refresh on first use, or when the working directory changed."""
        if not cls._cache or cls._scanned_cwd != Path.cwd():
            cls.refresh()

    @classmethod
    def names(cls) -> List[str]:
        """Return all available skill names (folder names or file stems)."""
        cls._ensure_loaded()

        return sorted(cls._cache.keys())

    @classmethod
    def exists(cls, skill_name: str) -> bool:
        """Check whether a skill exists."""
        cls._ensure_loaded()

        return skill_name in cls._cache

    @classmethod
    def load(cls, skill_name: str) -> Optional[str]:
        """Load a skill's content."""
        cls._ensure_loaded()

        return cls._cache.get(skill_name)

    @classmethod
    def load_many(cls, skill_names: List[str]) -> Dict[str, str]:
        """Load multiple skills at once."""
        cls._ensure_loaded()

        return {
            name: content
            for name, content in cls._cache.items()
            if name in skill_names
        }

    @classmethod
    def search(
        cls,
        query: str,
        *,
        search_content: bool = True,
    ) -> List[str]:
        """
        Search skills by name and optionally content.
        Returns matching skill names.
        """
        cls._ensure_loaded()

        query = query.lower().strip()
        matches = []

        for name, content in cls._cache.items():
            if query in name.lower():
                matches.append(name)
                continue

            if search_content and query in content.lower():
                matches.append(name)

        return sorted(matches)

    @classmethod
    def get_metadata(cls, skill_name: str) -> Optional[dict]:
        """Return basic skill metadata."""
        path = cls.path(skill_name)
        if not path or not path.exists():
            return None

        return {
            "name": skill_name,
            "path": str(path),
            "size": path.stat().st_size,
            "modified": path.stat().st_mtime,
        }

    @classmethod
    def path(cls, skill_name: str) -> Optional[Path]:
        """Return the file path for a skill."""
        cls._ensure_loaded()

        return cls._paths.get(skill_name)