# Drone-ViT-HW-Accelerator

## Set Up
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
