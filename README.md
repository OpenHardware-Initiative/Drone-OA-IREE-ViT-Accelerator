# Drone-ViT-HW-Accelerator

## Set Up

### 0 Create a `/data/` folder under the project root and copy the `environments.tar`, `flightrender.tar` and `pretrained_models.tar`
```bash
mkdir ./data/
cd data/
wget "https://docs.google.com/uc?export=download&confirm=t&id=1FWVKpyCuSvdDrPgkl5Ah7slsy19ZMIYu" -O pretrained_models.tar
wget "https://docs.google.com/uc?export=download&confirm=t&id=1iP9xxMWj8yz0kKL5WoGzjC55zz5OO-7A" -O flightrender.tar
wget "https://docs.google.com/uc?export=download&confirm=t&id=1axstq_ywY0KK5-2g-049o88qIf_rGYVq" -O environments.tar
```

### 1 Build and start the container
```bash
docker compose build
docker compose up -d
```

### 2 Attach to the running container
```bash
docker exec -it <name_of_container> bash
```
Or attach using VSCode devcontainers

### 3 Source ros workspace
```bash
cd ~/catkin_ws/
source devel/setup.bash
```
# 4 Run the sim
```bash
cd src/vitfly
bash launch_evaluation.bash 1 vision
```
