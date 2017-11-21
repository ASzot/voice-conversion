python convert.py \
--src p225 \
--trg p226 \
--model ConvVAE \
--checkpoint logdir/train/1119-1314-49-2017/model.ckpt-40931 \
--file_pattern "./dataset/vctk/bin/Training Set/{}/*.bin"
