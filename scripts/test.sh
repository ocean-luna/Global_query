bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    /home/ldaphome/seanl/code/liuxin/code/SparseDrive/work_dirs/sparsedrive_small_stage2_0909/latest.pth \
    1 \
    --deterministic \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl