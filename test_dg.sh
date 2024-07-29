bash tools/dist_test_dg.sh  \
DA/Ours/cityscapes/faster-rcnn_dift_fpn_cityscapes_source.py \
work_dirs/faster-rcnn_dift_fpn_cityscapes_source/iter_20000.pth \
8

bash tools/dist_test_dg.sh  \
DA/Ours/sim10k/faster-rcnn_dift_fpn_sim10k_source.py \
work_dirs/faster-rcnn_dift_fpn_sim10k_source/iter_20000.pth \
8

bash tools/dist_test_dg.sh  \
DA/Ours/voc/faster-rcnn_dift_fpn_voc_source.py \
work_dirs/faster-rcnn_dift_fpn_voc_source/iter_20000.pth \
8



#Folder="work_dirs"
#for file_name in ${Folder}/*
#do
#echo $file_name"/"${file_name##*/}".py"
#echo $file_name"/iter_20000.pth"
#bash tools/dist_test_dg.sh  $file_name"/"${file_name##*/}".py" $file_name"/iter_12000.pth" 8
#done

