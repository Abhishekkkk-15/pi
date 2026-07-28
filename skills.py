from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


class Skills:
    SKILLS_DIRS = [Path.cwd() / "skills", Path.cwd()]
    FILENAME = "SKILL.md"

    _cache: Dict[str, str] = {}
    _paths: Dict[str, Path] = {}

    @classmethod
    def refresh(cls) -> None:
        """Reload all skills into memory, supporting skills/<skill_name>/SKILL.md structure."""
        cls._cache.clear()
        cls._paths.clear()

        for base_dir in cls.SKILLS_DIRS:
            if not base_dir.exists():
                continue

            # 1. Folder structure: skills/<skill_name>/SKILL.md
            for folder in base_dir.iterdir():
                if folder.is_dir():
                    skill_name = folder.name
                    skill_file = folder / cls.FILENAME
                    if not skill_file.exists():
                        skill_file = folder / "skill.md"
                    if not skill_file.exists():
                        md_files = list(folder.glob("*.md"))
                        if md_files:
                            skill_file = md_files[0]

                    if skill_file.exists() and skill_file.is_file():
                        cls._cache[skill_name] = skill_file.read_text(encoding="utf-8")
                        cls._paths[skill_name] = skill_file

            # 2. Direct file structure: skills/<skill_name>.md
            for file in base_dir.glob("*.md"):
                if file.is_file() and file.stem.upper() != "SKILL":
                    skill_name = file.stem
                    if skill_name not in cls._cache:
                        cls._cache[skill_name] = file.read_text(encoding="utf-8")
                        cls._paths[skill_name] = file

    @classmethod
    def names(cls) -> List[str]:
        """Return all available skill names (folder names or file stems)."""
        if not cls._cache:
            cls.refresh()

        return sorted(cls._cache.keys())

    @classmethod
    def exists(cls, skill_name: str) -> bool:
        """Check whether a skill exists."""
        if not cls._cache:
            cls.refresh()

        return skill_name in cls._cache

    @classmethod
    def load(cls, skill_name: str) -> Optional[str]:
        """Load a skill's content."""
        if not cls._cache:
            cls.refresh()

        return cls._cache.get(skill_name)

    @classmethod
    def load_many(cls, skill_names: List[str]) -> Dict[str, str]:
        """Load multiple skills at once."""
        if not cls._cache:
            cls.refresh()

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
        if not cls._cache:
            cls.refresh()

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
        if not cls._cache:
            cls.refresh()

        return cls._paths.get(skill_name)