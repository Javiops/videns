"""Tiny Windows memory/process helpers shared by the stage-2 runner and builders (no torch import)."""
import ctypes
from ctypes import wintypes

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_INFORMATION = 0x0400
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
NORMAL_PRIORITY_CLASS = 0x20
ABOVE_NORMAL_PRIORITY_CLASS = 0x8000


class _MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_ram_bytes():
    m = _MS()
    m.dwLength = ctypes.sizeof(_MS)
    _kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.ullAvailPhys)


def set_priority_affinity(pids, priority_class, affinity_mask):
    """Re-schedule running processes (never touches what they compute)."""
    done = 0
    for pid in pids:
        h = _kernel32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, pid)
        if h:
            ok = _kernel32.SetPriorityClass(h, priority_class) and _kernel32.SetProcessAffinityMask(h, ctypes.c_size_t(affinity_mask))
            _kernel32.CloseHandle(h)
            done += bool(ok)
    return done
