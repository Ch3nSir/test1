"""Read and print to STDOUT DDP parameters (readable from bash)."""

import argparse
import os


def is_bolt() -> bool:
    """True if on bolt."""
    try:
        import turibolt as bolt

        instance_id = bolt.get_current_task_id()
        return instance_id is not None
    except Exception:
        pass

    return False


def is_irisctl() -> bool:
    """True if on irisctl."""
    try:
        from irisctl.api.runtime import SHARED_ARTIFACT_DIR

        return SHARED_ARTIFACT_DIR is not None
    except Exception:
        pass

    return False


def get_master_ip():
    """Grab the distributed master based on IRIS or bolt."""
    try:
        import turibolt as bolt  # noqa
        from irisctl.api.runtime import distributed_tasklets, role_rank
        for tasklet in distributed_tasklets():
            if tasklet.role_rank == 0:
                return tasklet.host_ip_address
    except Exception:
        pass

    distributed_master = "127.0.0.1"
    return distributed_master

def get_node_rank() -> int:
    """Grab the rank of the current node in the distributed process."""
    try:
        import turibolt as bolt  # noqa
        from irisctl.api.runtime import distributed_tasklets, role_rank
        if is_irisctl():
            return role_rank()
    except Exception:
        pass
    
    return 0


def get_master_port() -> int:
    """Grab the distributed master based on IRIS or bolt."""
    try:
        import turibolt as bolt  # noqa
        from irisctl.api.runtime import distributed_tasklets, role_rank
        for tasklet in distributed_tasklets():
            if tasklet.role_rank == 0:
                return tasklet.distributed_port
    except Exception:
        pass

    distributed_port = 29500
    return distributed_port


def get_world_size() -> int:
    """Return total world size for the parameters."""
    world_size = 1

    try:
        import turibolt as bolt  # noqa
        from irisctl.api.runtime import distributed_tasklets, role_rank

        # grab params and return the replicas which should be the world size
        parameters = bolt.get_current_config().get("parameters", None)
        if parameters is not None and "num_replicas" in parameters:
            return parameters["num_replicas"]

    except Exception as exception:
        if is_bolt():
            raise exception

    return world_size

def get_num_nodes() -> int:
    """Return total number of nodes for the parameters."""
    num_nodes = 1

    try:
        import turibolt as bolt  # noqa
        from irisctl.api.runtime import distributed_tasklets, role_rank

        # grab params and return the replicas which should be the world size
        parameters = bolt.get_current_config().get("parameters", None)
        if parameters is not None and "num_nodes" in parameters:
            return parameters["num_nodes"]

    except Exception as exception:
        if is_bolt():
            raise exception

    return num_nodes

def get_num_gpus() -> int:
    """Return number of GPUs per node."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
    except Exception:
        pass
    
    return 8  # Default to 8 GPUs per node

def get_standalone_str(
    pytorch_major_min: int = 1,
    pytorch_minor_min: int = 10,
) -> str:
    """Returns string for whether we are a standalone (single node) job.
    Only supported for PyTorch >= 1.10.0.
    """
    import torch  # noqa

    # also handle pip builds, eg: 1.11.0+cu113
    splits = torch.__version__.split(".")
    major = int(splits[0])
    minor = int(splits[1])
    standalone_supported = major >= pytorch_major_min and minor >= pytorch_minor_min

    standalone = get_world_size() == 1 and standalone_supported

    return f"{' --standalone' if standalone else ''}"


def get_uid() -> str:
    """Returns current GenCLR uid or default str."""
    uid = "compression_rag"

    try:
        import turibolt as bolt  # noqa

        # grab params and return the replicas which should be the world size
        parameters = bolt.get_current_config().get("parameters", None)
        if parameters is not None and "uid" in parameters:
            return parameters["uid"]

    except Exception as exception:
        if is_bolt():
            raise exception

    return uid


def is_genclr_entrypoint() -> bool:
    """Check if we are a GenCLR entry point or not."""
    is_genclr_flag = False

    # Check for a breakout environment var (set by evaluator/runner.py)
    if os.environ.get("EVALUATOR_OVERRIDE", False):
        return True

    try:
        import turibolt as bolt  # noqa

        # grab params and return the replicas which should be the world size
        parameters = bolt.get_current_config().get("parameters", None)
        if parameters is not None and "entry_point" in parameters:
            is_genclr_flag = parameters["entry_point"] == "genclr"
            return is_genclr_flag

    except Exception as exception:
        if is_bolt():
            raise exception

    return is_genclr_flag


if __name__ == "__main__":
    # build the mapper
    task_map = {
        "master": get_master_ip,
        "port": get_master_port,
        "rank": get_node_rank,
        "world_size": get_world_size,
        "uid": get_uid,
        "is_genclr_entrypoint": is_genclr_entrypoint,
        "standalone": get_standalone_str,
        "num_nodes": get_num_nodes,
        "num_gpus": get_num_gpus,
    }

    # parse args
    parser = argparse.ArgumentParser("Environment Parser.")
    parser.add_argument(
        "task",
        choices=tuple(task_map),
    )
    args = parser.parse_args()

    # Print to STDOUT.
    # Used to set a bash variable.
    print(task_map[args.task]()) 