"""Build the existing onedir spec without changing the installed interpreter."""
import dis
from pathlib import Path
import sys


def main():
    if sys.version_info[:3] == (3, 10, 0):
        # CPython bpo-45757: EXTENDED_ARG must reset on no-argument opcodes.
        # Build-process-only backport; https://bugs.python.org/issue45757
        def unpack(code):
            extended = 0
            for index in range(0, len(code), 2):
                opcode = code[index]
                if opcode >= dis.HAVE_ARGUMENT:
                    argument = code[index + 1] | extended
                    extended = argument << 8 if opcode == dis.EXTENDED_ARG else 0
                else:
                    argument, extended = None, 0
                yield index, opcode, argument
        dis._unpack_opargs = unpack
    import PyInstaller.__main__
    root = Path(__file__).resolve().parents[1]
    PyInstaller.__main__.run([str(root / "voxlux.spec"), *sys.argv[1:]])


if __name__ == "__main__":
    main()
