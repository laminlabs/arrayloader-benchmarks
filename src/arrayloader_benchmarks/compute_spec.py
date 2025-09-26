from __future__ import annotations

import json

import boto3


def get_aws_sagemaker_instance_type() -> str:
    """Get the instance type of the current SageMaker Studio instance."""
    try:
        # Read the metadata
        with open("/opt/ml/metadata/resource-metadata.json") as f:  # noqa
            metadata = json.load(f)

        sagemaker = boto3.client("sagemaker", region_name="us-west-2")

        # Try to describe the space
        space_response = sagemaker.describe_space(
            DomainId=metadata["DomainId"], SpaceName=metadata["SpaceName"]
        )

        # Navigate through the nested settings to find instance type
        space_settings = space_response.get("SpaceSettings", {})
        jupyter_settings = space_settings.get("JupyterLabAppSettings", {})
        default_resource_spec = jupyter_settings.get("DefaultResourceSpec", {})

        return default_resource_spec.get("InstanceType", "unknown")

    except Exception:  # noqa: BLE001
        return "unknown"
