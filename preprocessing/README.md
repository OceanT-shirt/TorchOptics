# Preprocessing

データの前処理用のライブラリです。

改善の余地はあると思うので、色々工夫をして頂けると嬉しく思います。

それぞれ、エンコーダー（学習用）と、デコーダー（検証用）がセットになっています。

## lens_sequence

レンズの構成及び、Aperture Stopの位置を扱います。

### Encoder: sequence_encoder

- Input:
  - t_all: **VARIABLE** length list of thickness between lens/pupil surfaces
  - stop_idx: stop index
  - t_length (param): the length of the output t. This should be larger than max(len(t_all))

- Output:
  - t: **FIXED** length thickness between ONLY lens surfaces
  - as_d: the distance between the aperture stop and the first lens surface

The length of output t is fixed, so it can be used for training.

The decoder (sequence_decoder) is Vice Versa.

### Example 1 (With Aperture Stop)

sequenceにAの重複がある場合（=Glass Surface間にAperture Stopが存在する場合）

```python
from preprocessing.lens_sequence import sequence_encoder, sequence_decoder

"""
Example: A_003
GAAGA
(0.019) 0.036 | 0.036 (0.019) 0.90
---
(): Lens surface
|: Pupil surface
numbers: thickness between surfaces
"""

# Encoder
t_all = [0.019, 0.036, 0.036, 0.019, 0.90]
is_independent_as = True  # GAAGA
stop_idx = 3
t_length = 6
t, as_d = sequence_encoder(t_all, is_independent_as, stop_idx, t_length)
print(t)
print(as_d)
```

Expected Output:
```
[0.019, 0.072, 0.019, 0.90,    0.,   0.   ]
0.055
```

### Example 2 (With Aperture Stop)

sequenceにAの重複がある場合（=Glass Surface間にAperture Stopが存在する場合）

かつ、Aが先頭に来る場合（= Aperture Stopが最初のレンズ表面より手前にある場合）

```python
from preprocessing.lens_sequence import sequence_encoder, sequence_decoder

"""
Example: D_006
AGAGA
| 15.24 (2.08) 23.56 (3.12) 25.4
---
(): Lens surface
|: Pupil surface
numbers: thickness between surfaces
"""

# Encoder
t_all = [15.24, 2.08, 23.56, 3.12, 25.4]
is_independent_as = True  # AGAGA
stop_idx = 1
t_length = 6
t, as_d = sequence_encoder(t_all, is_independent_as, stop_idx, t_length)
print(t)
print(as_d)
```

Expected Output:
```
[2.08, 23.56, 3.12, 25.4,    0.,   0.   ]
-15.24
```

### Example 3 (Without Aperture Stop)

sequenceにAの重複がなく、しかもAperture Stopが1である場合

```python
from preprocessing.lens_sequence import sequence_encoder, sequence_decoder

"""
Example: A_006
GAGA
(0.049) 0.074 (0.029) 0.79
---
(): Lens surface
|: Pupil surface
numbers: thickness between surfaces
"""

# Encoder
t_all = [0.049, 0.074, 0.029, 0.79]
is_independent_as = False  # GAGA
stop_idx = 1
t_length = 6
t, as_d = sequence_encoder(t_all, is_independent_as, stop_idx, t_length)
print(t)
print(as_d)
```

Expected Output:
```
[0.049, 0.074, 0.029, 0.79,    0.,   0.   ]
0.0
```

### Example 4 (Without Aperture Stop)

sequenceにAの重複がなく、しかもAperture Stopが1でない場合

```python
from preprocessing.lens_sequence import sequence_encoder, sequence_decoder

"""
Example: D_004
GAGA
(3.454) 89.48 |(1.270) 12.0
---
(): Lens surface
|: Pupil surface
numbers: thickness between surfaces
"""

# Encoder
t_all = [3.454, 89.48, 1.270, 12.0]
is_independent_as = False  # GAGA
stop_idx = 3
t_length = 6
t, as_d = sequence_encoder(t_all, is_independent_as, stop_idx, t_length)
print(t)
print(as_d)
```

Expected Output:
```
[3.454, 89.48, 1.270, 12.0,    0.,   0.   ]
92.934
```

## glass_material

### g_from_n_v

Converting nd, v into glass variables.



## utils

### calc_t_len

Calculate the t_length from the pd.Series of t_all. (Caluculate the max length of t_all)

### str2list

Convert the string of t_all or other columns loaded from csv files into the list of float.

# 余談

## sequence_decoderアルゴリズムについて

sequence_decoderは、Generatorが生成したあらゆる形式のデータをRay Tracing可能な形に変換する必要がある。

結果としてかなり複雑なアルゴリズムになっているので、以下のように整理する。


