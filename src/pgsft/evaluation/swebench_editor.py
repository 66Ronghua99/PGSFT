"""Standalone stdlib editor injected into benchmark containers."""

import json
import os
import sys

HISTORY_PATH = "/tmp/at_editor_history.json"


def logical_lines(content):
    lines = content.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def cat_n(path, view_range=None):
    with open(path, encoding="utf-8") as handle:
        lines = logical_lines(handle.read())
    start, end = 1, len(lines)
    if view_range is not None:
        start, end = view_range
        if start < 1 or start > len(lines) or (end != -1 and (end < start or end > len(lines))):
            raise ValueError("view_range is out of range for file with %d lines" % len(lines))
        if end == -1:
            end = len(lines)
    return "\n".join("%6d\t%s" % (i, line) for i, line in enumerate(lines[start - 1:end], start))


def load_history():
    try:
        with open(HISTORY_PATH, encoding="utf-8") as handle:
            history = json.load(handle)
    except FileNotFoundError:
        return {}
    if not isinstance(history, dict) or any(
        not isinstance(v, list) or any(x is not None and not isinstance(x, str) for x in v)
        for v in history.values()
    ):
        raise ValueError("Invalid editor history")
    return history


def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as handle:
        json.dump(history, handle)


def validate(data):
    if not isinstance(data, dict):
        raise ValueError("arguments must be an object")
    cmd = data.get("command")
    if not isinstance(cmd, str) or cmd not in ("view", "create", "str_replace", "insert", "undo_edit"):
        raise ValueError("Unknown command: %s" % cmd)
    if not isinstance(data.get("path"), str) or not data["path"]:
        raise ValueError("path must be a non-empty string")
    required = {"create": ("file_text",), "str_replace": ("old_str",), "insert": ("new_str",)}
    for key in required.get(cmd, ()):
        if not isinstance(data.get(key), str):
            raise ValueError("%s requires string %s" % (cmd, key))
    if cmd == "str_replace":
        if not data["old_str"]:
            raise ValueError("old_str must not be empty")
        if "new_str" in data and not isinstance(data["new_str"], str):
            raise ValueError("new_str must be a string")
    if cmd == "insert" and type(data.get("insert_line")) is not int:
        raise ValueError("insert_line must be an integer")
    if cmd == "view" and "view_range" in data:
        value = data["view_range"]
        if not isinstance(value, list) or len(value) != 2 or any(type(x) is not int for x in value):
            raise ValueError("view_range must be [start, end] integers")
    return cmd, os.path.abspath(data["path"])


def edit(data):
    cmd, path = validate(data)
    if cmd == "view":
        if os.path.isdir(path):
            if "view_range" in data:
                raise ValueError("view_range is not supported for directories")
            out = [path + "/"]
            for entry in sorted(os.listdir(path)):
                if entry.startswith("."):
                    continue
                child = os.path.join(path, entry)
                out.append(child + "/" if os.path.isdir(child) else child)
                if os.path.isdir(child):
                    for sub in sorted(os.listdir(child)):
                        if not sub.startswith("."):
                            nested = os.path.join(child, sub)
                            out.append(nested + "/" if os.path.isdir(nested) else nested)
            return "Here's the files and directories up to 2 levels deep in %s, excluding hidden items:\n%s" % (path, "\n".join(out))
        return "Here's the result of running `cat -n` on %s:\n%s" % (path, cat_n(path, data.get("view_range")))

    history = load_history()
    if cmd == "undo_edit":
        stack = history.get(path, [])
        if not stack:
            raise ValueError("No edit history found for %s." % path)
        previous = stack[-1]
        if previous is None:
            os.unlink(path)
            message = "File creation undone at: %s." % path
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(previous)
            message = "The file %s has been reverted to its previous state.\n%s" % (path, cat_n(path))
        stack.pop()
        save_history(history)
        return message

    if cmd == "create":
        if os.path.lexists(path):
            raise ValueError("File already exists at: %s. Cannot overwrite files using command `create`." % path)
        previous, updated = None, data["file_text"]
    else:
        with open(path, encoding="utf-8") as handle:
            previous = handle.read()
        if cmd == "str_replace":
            count = previous.count(data["old_str"])
            if count != 1:
                raise ValueError("old_str must appear exactly once in %s (found %d matches)." % (path, count))
            updated = previous.replace(data["old_str"], data.get("new_str", ""), 1)
        else:
            lines = logical_lines(previous)
            index = data["insert_line"]
            if not 0 <= index <= len(lines):
                raise ValueError("Insert line %s is out of range for file %s (has %d lines)." % (index, path, len(lines)))
            lines[index:index] = logical_lines(data["new_str"])
            updated = "\n".join(lines)
            if previous.endswith("\n") or data["new_str"].endswith("\n"):
                updated += "\n"
    with open(path, "x" if cmd == "create" else "w", encoding="utf-8") as handle:
        handle.write(updated)
    history.setdefault(path, []).append(previous)
    history[path] = history[path][-10:]
    save_history(history)
    prefix = "File created successfully at: %s." if cmd == "create" else "The file %s has been edited."
    return (prefix % path) + " Here's the result of running `cat -n` on the file: %s:\n%s" % (path, cat_n(path))


def main():
    try:
        print(edit(json.load(sys.stdin)))
    except (ValueError, OSError, TypeError) as error:
        print("Editor error: %s" % error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
