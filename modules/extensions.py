import os
import threading

from modules import cache, errors, shared
from modules.paths_internal import script_path  # noqa: F401
from modules.paths_internal import extensions_builtin_dir, extensions_dir

extensions = []

os.makedirs(extensions_dir, exist_ok=True)


def active():
    if shared.opts.disable_all_extensions == "all":
        return []
    elif shared.opts.disable_all_extensions == "extra":
        return [x for x in extensions if x.enabled and x.is_builtin]
    else:
        return [x for x in extensions if x.enabled]


class Extension:
    lock = threading.Lock()
    cached_fields = ["remote", "commit_date", "branch", "commit_hash", "version"]

    def __init__(self, name, path, enabled=True, is_builtin=False):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.status = ""
        self.can_update = False
        self.is_builtin = is_builtin
        self.commit_hash = ""
        self.commit_date = None
        self.version = ""
        self.branch = None
        self.remote = None
        self.have_info_from_repo = False

    def to_dict(self):
        return {x: getattr(self, x) for x in self.cached_fields}

    def from_dict(self, d):
        for field in self.cached_fields:
            setattr(self, field, d[field])

    def read_info_from_repo(self):
        if self.is_builtin or self.have_info_from_repo:
            return

        def read_from_repo():
            with self.lock:
                if self.have_info_from_repo:
                    return

                self.do_read_info_from_repo()

                return self.to_dict()

        try:
            d = cache.cached_data_for_file(
                "extensions-git",
                self.name,
                os.path.join(self.path, ".git"),
                read_from_repo,
            )
            self.from_dict(d)
        except FileNotFoundError:
            pass
        self.status = "unknown" if self.status == "" else self.status

    def do_read_info_from_repo(self):
        repo = None
        try:
            if os.path.exists(os.path.join(self.path, ".git")):
                repo = Repo(self.path)
        except Exception:
            errors.report(
                f"Error reading github repository info from {self.path}", exc_info=True
            )

        if repo is None or repo.bare:
            self.remote = None
        else:
            try:
                self.remote = next(repo.remote().urls, None)
                commit = repo.head.commit
                self.commit_date = commit.committed_date
                if repo.active_branch:
                    self.branch = repo.active_branch.name
                self.commit_hash = commit.hexsha
                self.version = self.commit_hash[:8]

            except Exception:
                errors.report(
                    f"Failed reading extension data from Git repository ({self.name})",
                    exc_info=True,
                )
                self.remote = None

        self.have_info_from_repo = True

    def list_files(self, subdir, extension):
        from modules import scripts

        dirpath = os.path.join(self.path, subdir)
        if not os.path.isdir(dirpath):
            return []

        res = []
        for filename in sorted(os.listdir(dirpath)):
            res.append(
                scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename))
            )

        res = [
            x
            for x in res
            if os.path.splitext(x.path)[1].lower() == extension
            and os.path.isfile(x.path)
        ]

        return res


def list_extensions():
    extensions.clear()

    if not os.path.isdir(extensions_dir):
        return

    if shared.opts.disable_all_extensions == "all":
        print(
            '*** "Disable all extensions" option was set, will not load any extensions ***'
        )
    elif shared.opts.disable_all_extensions == "extra":
        print(
            '*** "Disable all extensions" option was set, will only load built-in extensions ***'
        )

    extension_paths = []
    for dirname in [extensions_dir, extensions_builtin_dir]:
        if not os.path.isdir(dirname):
            return

        for extension_dirname in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, extension_dirname)
            if not os.path.isdir(path):
                continue

            extension_paths.append(
                (extension_dirname, path, dirname == extensions_builtin_dir)
            )

    for dirname, path, is_builtin in extension_paths:
        extension = Extension(
            name=dirname,
            path=path,
            enabled=dirname not in shared.opts.disabled_extensions,
            is_builtin=is_builtin,
        )
        extensions.append(extension)
