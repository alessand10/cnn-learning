# Handle to the workspace
from azure.ai.ml import MLClient

# Authentication package
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential

from azure.ai.ml.entities import AmlCompute

credential = InteractiveBrowserCredential(tenant_id='0a37665b-0281-435e-b91c-b08fd4915ab6')

ml_client = MLClient(
    credential=credential,
    subscription_id="b33e645c-f2fe-44b5-9f89-748adb0757e2",
    resource_group_name="NN",
    workspace_name="TestWorkspace",
)

gpu_compute_target = "gpu-cluster"


'''Create Compute Resource'''
try:
    # let's see if the compute target already exists
    gpu_cluster = ml_client.compute.get(gpu_compute_target)
    print(
        f"You already have a cluster named {gpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new gpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    gpu_cluster = AmlCompute(
        # Name assigned to the compute cluster
        name="gpu-cluster",
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_NC6",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )

    # Now, we pass the object to MLClient's create_or_update method
    gpu_cluster = ml_client.begin_create_or_update(gpu_cluster).result()

print(
    f"AMLCompute with name {gpu_cluster.name} is created, the compute size is {gpu_cluster.size}"
)

'''Create Job Environment for job'''

curated_env_name = "AzureML-tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu@latest"
web_path = "wasbs://datasets@azuremlexamples.blob.core.windows.net/mnist/"



from azure.ai.ml import command
from azure.ai.ml import UserIdentityConfiguration
from azure.ai.ml import Input
''' Build the job '''

job = command(
    compute=gpu_compute_target,
    environment=curated_env_name,
    code="./",
    command="python gan",
    experiment_name="tf-dnn-image-classify",
    display_name="simple-nn",
)

''' Submit the job '''
ml_client.jobs.create_or_update(job)

a=5
