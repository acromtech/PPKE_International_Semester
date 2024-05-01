[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/aqpRtTM8)

## To compile
```
source /project/p_covidpre/source_libs
nvcc <file.cu> -o <executable> -arch=sm_80
./<executable>
```

```
srun -p gpu --gres=gpu:1 --ntasks=1 --time=00:05:00 --mem=40G --reservation=p_covidpre_123 ./first
```