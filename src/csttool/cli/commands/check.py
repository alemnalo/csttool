import argparse
import sys
from .doctor import cmd_doctor


def cmd_check(args: argparse.Namespace) -> bool:
    """Deprecated: redirects to cmd_doctor."""
    print("Note: 'csttool check' is deprecated — use 'csttool doctor' instead.", file=sys.stderr)
    return cmd_doctor(args)
