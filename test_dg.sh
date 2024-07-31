bash tools/dist_test_dg.sh  \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu.pyz \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e/iter_20000.pth \
8

bash tools/dist_test_dg.sh  \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu.py \
work_dirs/faster-rcnn_r101+dift_fpn_dwd_semi_base_e2e_albu/iter_20000.pth \
8

#Folder="work_dirs"
#for file_name in ${Folder}/*
#do
#echo $file_name"/"${file_name##*/}".py"
#echo $file_name"/iter_20000.pth"
#bash tools/dist_test_dg.sh  $file_name"/"${file_name##*/}".py" $file_name"/iter_12000.pth" 8
#done

