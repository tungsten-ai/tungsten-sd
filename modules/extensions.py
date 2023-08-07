import os
import threading

from modules import shared
from modules.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401

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

    def __init__(self, name, path, enabled=True, is_builtin=False):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.status = ''
        self.can_update = False
        self.is_builtin = is_builtin
        self.commit_hash = ''
        self.commit_date = None
        self.version = ''
        self.branch = None
        self.remote = None

    def list_files(self, subdir, extension):
        from modules import scripts

        dirpath = os.path.join(self.path, subdir)
        if not os.path.isdir(dirpath):
            return []

        res = []
        for filename in sorted(os.listdir(dirpath)):
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename)))

        res = [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

        return res


def list_extensions():
    extensions.clear()

    if not os.path.isdir(extensions_dir):
        return

    if shared.opts.disable_all_extensions == "all":
        print("*** \"Disable all extensions\" option was set, will not load any extensions ***")
    elif shared.opts.disable_all_extensions == "extra":
        print("*** \"Disable all extensions\" option was set, will only load built-in extensions ***")

    extension_paths = []
    for dirname in [extensions_dir, extensions_builtin_dir]:
        if not os.path.isdir(dirname):
            return

        for extension_dirname in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, extension_dirname)
            if not os.path.isdir(path):
                continue

            extension_paths.append((extension_dirname, path, dirname == extensions_builtin_dir))

    for dirname, path, is_builtin in extension_paths:
        extension = Extension(name=dirname, path=path, enabled=dirname not in shared.opts.disabled_extensions, is_builtin=is_builtin)
        extensions.append(extension)
