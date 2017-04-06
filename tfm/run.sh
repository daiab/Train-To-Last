export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p log/runtime
mkdir -p log/train
`CUDA_VISIBLE_DEVICES=0 python release/googlenet/main/train.py --job_name=worker --task_index=0 > log/runtime/tmp_0.log 2>&1 &`
`CUDA_VISIBLE_DEVICES=1 python release/googlenet/main/train.py --job_name=worker --task_index=1 > log/runtime/tmp_1.log 2>&1 &`
`CUDA_VISIBLE_DEVICES=2 python release/googlenet/main/train.py --job_name=worker --task_index=2 > log/runtime/tmp_2.log 2>&1 &`
`CUDA_VISIBLE_DEVICES=3 python release/googlenet/main/train.py --job_name=worker --task_index=3 > log/runtime/tmp_3.log 2>&1 &`
`CUDA_VISIBLE_DEVICES="" python release/googlenet/main/train.py --job_name=ps --task_index=0 > log/runtime/tmp_cpu.log 2>&1 &`

# `CUDA_VISIBLE_DEVICES=0 python release/googlenet/main/train.py --job_name=worker --task_index=4 > log/runtime/tmp_0.log 2>&1 &`
# `CUDA_VISIBLE_DEVICES=1 python release/googlenet/main/train.py --job_name=worker --task_index=5 > log/runtime/tmp_1.log 2>&1 &`
# `CUDA_VISIBLE_DEVICES=2 python release/googlenet/main/train.py --job_name=worker --task_index=6 > log/runtime/tmp_2.log 2>&1 &`
# `CUDA_VISIBLE_DEVICES=3 python release/googlenet/main/train.py --job_name=worker --task_index=7 > log/runtime/tmp_3.log 2>&1 &`
# `CUDA_VISIBLE_DEVICES="" python release/googlenet/main/train.py --job_name=ps --task_index=0 > log/runtime/tmp_cpu.log 2>&1 &`
