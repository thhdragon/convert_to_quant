"""
parallel_utils - Multi-GPU parallel execution utilities for layer quantization.
"""

import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Union
import torch

from .logging import info, verbose, warning


def parse_devices(
    device: Optional[str] = None,
    devices: Optional[Union[str, List[str]]] = None,
    num_gpus: Optional[int] = None,
) -> List[str]:
    """
    Parse device CLI parameters into a clean list of PyTorch device strings.

    Args:
        device: Single device string (e.g. 'cuda', 'cuda:0', 'cpu', or comma-separated 'cuda:0,cuda:1')
        devices: Devices string or list (e.g. 'cuda:0,cuda:1' or ['cuda:0', 'cuda:1'] or '0,1')
        num_gpus: Integer count of GPUs to use (e.g. 2 -> ['cuda:0', 'cuda:1'])

    Returns:
        List of device strings, e.g. ['cuda:0', 'cuda:1'] or ['cuda'] or ['cpu']
    """
    # 1. Check explicit devices parameter
    if devices:
        if isinstance(devices, str):
            parts = [p.strip() for p in devices.replace(",", " ").split() if p.strip()]
        else:
            parts = [str(p).strip() for p in devices if str(p).strip()]

        dev_list = []
        for p in parts:
            if p.isdigit():
                dev_list.append(f"cuda:{p}")
            elif not p.startswith("cuda") and p != "cpu":
                dev_list.append(f"cuda:{p}")
            else:
                dev_list.append(p)
        if dev_list:
            return dev_list

    # 2. Check explicit num_gpus parameter
    if num_gpus is not None and num_gpus > 0:
        if torch.cuda.is_available():
            available = torch.cuda.device_count()
            count = min(num_gpus, available) if available > 0 else num_gpus
            return [f"cuda:{i}" for i in range(count)]
        return ["cpu"]

    # 3. Check single device string (might contain comma/space)
    if device:
        if "," in device or " " in device:
            parts = [p.strip() for p in device.replace(",", " ").split() if p.strip()]
            dev_list = []
            for p in parts:
                if p.isdigit():
                    dev_list.append(f"cuda:{p}")
                elif not p.startswith("cuda") and p != "cpu":
                    dev_list.append(f"cuda:{p}")
                else:
                    dev_list.append(p)
            if dev_list:
                return dev_list
        return [device]

    # 4. Default auto-detect
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    return [default_dev]


def run_parallel_layer_processing(
    work_items: List[Any],
    process_fn: Callable[[Any, str], Dict[str, Any]],
    devices: List[str],
) -> List[Dict[str, Any]]:
    """
    Process work_items across multiple GPU devices in parallel using ThreadPoolExecutor.

    Args:
        work_items: List of items to process (e.g. (index, key))
        process_fn: Function taking (item, device_str) and returning a result dict
        devices: List of CUDA or CPU device strings (e.g. ['cuda:0', 'cuda:1'])

    Returns:
        List of result dicts from process_fn
    """
    if not work_items:
        return []

    # If single device, execute sequentially
    if not devices or len(devices) <= 1:
        single_dev = devices[0] if devices else ("cuda" if torch.cuda.is_available() else "cpu")
        results = []
        for item in work_items:
            results.append(process_fn(item, single_dev))
        return results

    info(f"Multi-GPU parallel quantization enabled across {len(devices)} GPUs: {', '.join(devices)}")

    device_queue: queue.Queue = queue.Queue()
    for dev in devices:
        device_queue.put(dev)

    def worker_task(item: Any) -> Dict[str, Any]:
        dev = device_queue.get()
        try:
            if str(dev).startswith("cuda"):
                dev_obj = torch.device(dev)
                with torch.cuda.device(dev_obj):
                    res = process_fn(item, dev)
                    torch.cuda.empty_cache()
                    return res
            else:
                return process_fn(item, dev)
        finally:
            device_queue.put(dev)

    results = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker_task, item) for item in work_items]
        for future in as_completed(futures):
            results.append(future.result())

    return results
