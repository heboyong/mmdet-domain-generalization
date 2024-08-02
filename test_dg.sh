#bash tools/dist_test_dg.sh  \
#work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_-1/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_-1.py \
#work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_-1/iter_20000.pth \
#8
#
#bash tools/dist_test_dg.sh  \
#work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_0.0004/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_0.0004.py \
#work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_0.0004/iter_20000.pth \
#8

bash tools/dist_test_dg.sh  \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_mse/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_mse.py \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_mse/iter_8000.pth \
8

bash tools/dist_test_dg.sh  \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_mse/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_mse.py \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_mse/iter_12000.pth \
8

#Folder="work_dirs"
#for file_name in ${Folder}/*
#do
#echo $file_name"/"${file_name##*/}".py"
#echo $file_name"/iter_20000.pth"
#bash tools/dist_test_dg.sh  $file_name"/"${file_name##*/}".py" $file_name"/iter_12000.pth" 8
#done

