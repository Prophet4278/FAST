CUDA_VISIBLE_DEVICES=0 \
python test.py \
     --eval-only \
     --num-gpus 1 \
     --config /home/user/zhangdan/code/CMT-main/CMT_PT/configs/pt/final_c2f_0.02.yaml\
      MODEL.WEIGHTS /home/user/zhangdan/code/CMT-main/CMT_PT/output/c2f_10/model_0029999.pth\
      MODEL.ANCHOR_GENERATOR.NAME "DifferentiableAnchorGenerator" \
      UNSUPNET.EFL True \
      UNSUPNET.EFL_LAMBDA [0.5,0.5] \
      UNSUPNET.TAU [0.5,0.5]