# Preprocess

| major_only | method       | pad_global | Feature_Dimension | Detail                                                       |
| ---------- | ------------ | ---------- | ----------------- | ------------------------------------------------------------ |
| -1         | any          | False      | 6                 | Variance Higher + Variance Lower if exists                   |
| 3          | any          | False      | 6                 | Body_1 + Body_2 if exists                                    |
| 0/1/2      | none         | False      | 3                 | Body only  (0:Variance Higher; 1:Body_1; 2:Body_2 if exists) |
| 0/1/2      | max/min/mean | True       | 6                 | Body+Global Pooling                                          |
