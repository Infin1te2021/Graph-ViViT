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

## Results (Masked)

| Type | 60CS | 60CV | 120CSub | 120CSet |
| --- | ---| --- | --- | --- |
| Full Masked | 86.20 | 89.67* | 81.70 | 83.73 |
| Spatial Masked | 86.27 | 88.89* | 81.17 | 79.87* |

\* Denotes early stopped.