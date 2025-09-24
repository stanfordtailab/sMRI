import os, re, glob
from typing import Iterator, Tuple, Optional

def _to_posix(p: str) -> str:
    return p.replace("\\", "/")

def structure_to_regex(structure: str, root: str) -> re.Pattern:
    """Convert structure pattern to regex with named groups for subject/session."""
    s = _to_posix(structure)
    r = _to_posix(root.rstrip("/"))
    # Replace root (fixed value)
    s = s.replace("{root}", re.escape(r))
    # Convert wildcards
    s = s.replace("**", ".*")         # Any number of levels
    s = s.replace("*",  "[^/]*")      # Within one level
    # Convert placeholders
    s = s.replace("{subject}", r"(?P<subject>[^/]+)")
    s = s.replace("{session}", r"(?P<session>[^/]+)")
    # Start/end anchors
    return re.compile("^" + s.rstrip("/") + "/?$")

def structure_to_glob(structure: str, root: str, subjects=None, sessions=None) -> str:
    """Glob pattern for listing candidates (subjects/sessions replaced with * if None)."""
    s = _to_posix(structure)
    s = s.replace("{root}", _to_posix(root.rstrip("/")))
    s = s.replace("{subject}", subjects if isinstance(subjects, str) else "*")
    s = s.replace("{session}", sessions if isinstance(sessions, str) else "*")
    return s

def discover(structure: str, root: str,
             subjects=None, sessions=None) -> Iterator[Tuple[str, Optional[str], str]]:
    """
    Find paths matching the pattern and return (subject, session, path).
    If {session} is not in pattern, session=None is returned.
    """
    has_session = "{session}" in structure
    subj_list = subjects if isinstance(subjects, list) else [subjects] if isinstance(subjects, str) else [None]
    sess_list = sessions if (has_session and isinstance(sessions, list)) else [sessions] if (has_session and isinstance(sessions, str)) else [None]

    regex = structure_to_regex(structure, root)
    seen = set()

    for subj in subj_list:
        for sess in (sess_list if has_session else [None]):
            g = structure_to_glob(structure, root,
                                  subjects=subj if isinstance(subj, str) else None,
                                  sessions=sess if isinstance(sess, str) else None)
            for path in glob.glob(g, recursive=True):
                p = _to_posix(os.path.abspath(path)).rstrip("/")
                m = regex.match(p)
                if not m: 
                    continue
                subject = m.groupdict().get("subject")
                session = m.groupdict().get("session") if has_session else None
                seen.add((subject, session, p))

    for subject, session, p in sorted(seen, key=lambda x: (x[0], x[1] or "", x[2])):
        yield subject, session, p
                
def make_path(structure: str, root: str, subject: str, session: Optional[str], modality: Optional[str] = None, mkdirs: bool=False) -> str:
    """Generate output path for a subject/session/modality."""
    s = structure.replace("{root}", root.rstrip("/")).replace("{subject}", subject)
    if "{session}" in s:
        s = s.replace("{session}", session or "").replace("//", "/")
    if "{modality}" in s:
        s = s.replace("{modality}", modality or "").replace("//", "/")
    s = s.rstrip("/")
    if mkdirs:
        os.makedirs(s, exist_ok=True)
    return s

