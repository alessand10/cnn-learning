# Handle to the workspace
from azure.ai.ml import MLClient

# Authentication package
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential

from azure.ai.ml.entities import AmlCompute

credential = InteractiveBrowserCredential(tenant_id='0a37665b-0281-435e-b91c-b08fd4915ab6')
sub_id = "b33e645c-f2fe-44b5-9f89-748adb0757e2"
ws_name = "TestWorkspace"


ml_client = MLClient(
    credential=credential,
    subscription_id=sub_id,
    resource_group_name="NN",
    workspace_name=ws_name,
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



from azureml.core.environment import CondaDependencies
from azureml.core import Environment

from azureml.core import Workspace

ws = Workspace.get(
    name=ws_name,
    subscription_id=sub_id,
    resource_group='NN',
)




myenv = Environment(name='MyEnv90')
conda_dep = CondaDependencies()
conda_dep.add_conda_package('pip')
conda_dep.add_pip_package('numpy==1.22.0')
conda_dep.add_pip_package('tensorflow==2.12.0')
conda_dep.add_pip_package('opencv-python==4.7.0.72')
conda_dep.add_pip_package('matplotlib==3.7.1')
myenv.docker.base_dockerfile = './Dockerfile'
myenv.docker.base_image = None
myenv.python.conda_dependencies=conda_dep
myenv.version='1'
envObj = myenv.register(ws)



from azureml.core import ScriptRunConfig, Experiment
from azure.ai.ml import command

myexp = Experiment(workspace=ws, name = "environment-example")
src = ScriptRunConfig(source_directory="./",
                      script="gan_test.py",
                      compute_target=ws.compute_targets['simple'],
                      environment=myenv)

run = myexp.submit(config=src)
run.wait_for_completion(show_output=True)

a=5
