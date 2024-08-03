bash tools/dist_test_dg.sh  \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_cross/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_cross.py \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_cross/iter_20000.pth \
8

bash tools/dist_test_dg.sh  \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_mse_cross/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_mse_cross.py \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu_mse_cross/iter_20000.pth \
8

bash tools/dist_test_dg.sh  \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_nosemi/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_nosemi.py \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_nosemi/iter_20000.pth \
8

#Folder="work_dirs"
#for file_name in ${Folder}/*
#do
#echo $file_name"/"${file_name##*/}".py"
#echo $file_name"/iter_20000.pth"
#bash tools/dist_test_dg.sh  $file_name"/"${file_name##*/}".py" $file_name"/iter_12000.pth" 8
#done

