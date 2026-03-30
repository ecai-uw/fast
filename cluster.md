## Step 0: Dockerize the Project

Create a docker image within which to run any training code. Look at Dockerfile in this repo for reference, as well as the following `docker run` command as a reference to debug things/make sure the container works as desired.
```
sudo docker run --rm --runtime=nvidia -t -e WANDB_MODE=disabled -e WANDB_API_KEY=$WANDB_API_KEY -e PYTHONUNBUFFERED=1 -v ./:/opt/code/fast/ --gpus all fast python train_fast.py --config-path=cfg/robomimic --config-name=fast_can.yaml load_offline_data=True policy.shape_rewards=True env.reward_offset=0 env.n_envs=1 env.n_eval_envs=1
```
Some minor context for the command above:
 - `-e PYTHONUNBUFFERED=1` prevents the Docker image from buffering any logged Python outputs (very helpful for debugging/test runs).
 - `-v ./:/opt/code/fast/` mounts the current directory (assuming the Docker image is being run from the project directory) to the working directory inside the container. The specific paths will depend on the project hierarchy *and* the container hierarchy (in other words, dependent on the Dockerfile), but the overall point of this mounting is to allow the container to access up-to-date code without re-building the image (very slow). Apptainer/Singularity allow for a similar functionality.


## Step 0.5: Setup Miscellanea

Pre-empting some small considerations that will likely come up in future steps:
- **[WandB setup]**: if the project logs to WandB, add WANDB_API_KEY as an environment variable to the cluster account so that the job script can easily send the WANDB_API_KEY to the container itself.
- Get familiar with storage hierarchy: where will projects/code/data/logs/containers go? This is very cluster dependent, and it's worth sorting out now to crystallize all of the paths in the next steps.


## Step 1: Apptainer-ify the Project Image

The cluster most likely uses Apptainer instead of Docker. As such, we'll need to convert the Docker image into an Apptainer/Singularity file:
```
apptainer pull /path/to/save/apptainer/image docker::/user/repo[:tagname]
```
**NOTE:** this should probably be done locally (will require installing Apptainer locally), and then transfered to the cluster to avoid IO issues with the following command:
```
rsync -avPn /path/to/sif user@cluster:/desired/storage/location
```
Remove the `-n` flag to turn off dry-run.

## Step 2: Run the Apptainer Image



## Step 3: Copy Code to Cluster
```
rsync -avPn --exclude="wandb/*" --exclude="logs/*" --exclude="debug/*" --exclude=".git/*" /project/dir/ user@cluster:/desired/project/location
```
What I run for this project for Klone:
```
rsync -avPn --exclude="wandb/*" --exclude="logs/*" --exclude=".git/*" --exclude="debug/*" ~/fast/ ecai0608@klone.hyak.uw.edu:/gscratch/weirdlab/ecai0608/fast_project/fast/
```
and for Tillicum:
```
rsync -avPn --exclude="wandb/*" --exclude="logs/*" --exclude=".git/*" --exclude="debug/*" ~/fast/ ecai0608@tillicum.hyak.uw.edu:/gpfs/scrubbed/ecai0608/fast_project/fast/
```


sbatch scripts/launch_tillicum.slurm online robomimic_can policy.type=residual policy.shape_rewards=True

sbatch scripts/launch_tillicum.slurm online policy.type=residual policy.shape_rewards=True policy.residual_mag=1.0 policy.base_gradient_steps=-1

sbatch scripts/launch_tillicum.slurm online policy.type=residual_scale2 policy.shape_rewards=True policy.residual_mag=1.0 policy.base_gradient_steps=-1

sbatch scripts/launch_tillicum.slurm online policy.type=residual_force2 policy.shape_rewards=True policy.residual_mag=1.0 policy.base_gradient_steps=-1