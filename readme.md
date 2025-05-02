# Graph-ViViT

## Run

```bash
python train.py --config config.json
```

## Resume

```bash
python train.py --resume path/to/checkpoint
```

## Test

```bash
python test.py --resume path/to/checkpoint
```

## Vis

```bash
tensorboard --logdir saved/log/
```

## Data Preparation

### Directory Structure

Unzip NTU-RGB+D 60 and 120 skeleton datasets to the following directory structure:

```text
- data
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgbd_skeletons/      # from `nturgbd_skeletons_s001_to_s017.zip`
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
```

- Run the following command:

```bash
cd ./data/ntu # or cd ./data/ntu120
# Get skeleton of each performer
python get_raw_skes_data.py
# Remove the bad skeleton 
python get_raw_denoised_data.py
# Transform the skeleton to the center of the first frame
python seq_transformation.py
```

## Results (Abaltion Study on Embedding)

| Type | 60CS | 60CV | 120CSub | 120CSet |
| --- | ---| --- | --- | --- |
| Node | 86.20 | 89.67* | 81.70 | 83.73 |
| Velocity | 86.52 | 92.13* | 81.93 | 83.44 |
| Centrality | 81.21 | 86.15* | 73.96 | 0.83* (69) |
| Node + Velocity | 86.68* | 91.97* | 82.57 | - |
| Node + Centrality | 86.12 | 91.22 | 80.99 | 82.84 |
| Velocity + Centrality | - | - | - | - |

\* Denotes early stopped.